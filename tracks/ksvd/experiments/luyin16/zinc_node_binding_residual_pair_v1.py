"""ZINC node-binding residual pair v1 (2026-10-02).

One frozen question: starting from the *same* canonical fresh Full
initialisation and the same complete 240-epoch protocol, does replacing the
pure multiplicative node binding with a **fixed-amplitude cross-term + additive
residual** keep the node channels learnable and change unseen-graph error?

Arms (identical protocol, config-only difference):
  ``control``   — original ``u0 = (s * a) / sqrt(D_A)``
  ``residual``  — ``u1 = gamma * [u0 + eta_s * s + eta_a * a]``

``eta_s``, ``eta_a``, ``gamma`` are three fixed, non-trainable buffers computed
once from the canonical shared initial state on a fixed random 1024-graph train
sample.  No new trainable parameter, no LayerNorm / gate / activation / extra
loss.  ``node_binding_zero`` still zeroes the whole ``u`` (additive terms
included).  Official ZINC **test** is never instantiated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1lib
from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as uprun
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = "zinc-node-binding-residual-pair-v1"
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
EXPORT_DIR = TRACK_ROOT / "results/zinc_node_binding_residual_pair_v1"

SEED = 0
SCALE_SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
DIAG_EPOCHS = (0, 1, 10, 40, 80, 160, 240)
COEFF_GRAPHS = 1024
DIAG_BATCH_SIZE = 128

ARMS = ("control", "residual")

_write_json = uprun._write_json
_read_json = uprun._read_json


# ---------------------------------------------------------------------------
# seeding / model
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class ResidualNodeBindingScaleModel(sc.LatentScaleSEM108):
    """Full scale model whose node binding is the fixed-amplitude residual.

    All original parameters and the RNG stream are inherited unchanged; the
    only addition is three non-persistent fixed scalar buffers.  ``eta_s =
    eta_a = 0`` and ``gamma = 1`` reduce the binding to the frozen product.
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        eta_s: float = 0.0,
        eta_a: float = 0.0,
        gamma: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, dictionary, subspace=subspace, **kwargs)
        self.register_buffer("eta_s", torch.tensor(float(eta_s)), persistent=False)
        self.register_buffer("eta_a", torch.tensor(float(eta_a)), persistent=False)
        self.register_buffer("gamma", torch.tensor(float(gamma)), persistent=False)
        self.binding_mode = "residual"

    def _node_binding(self, c: torch.Tensor, qc: torch.Tensor) -> torch.Tensor:
        s = c @ self.W_A_S
        a = qc @ self.W_A_C
        u0 = (s * a) / math.sqrt(float(p2.D_A))
        return self.gamma * (u0 + self.eta_s * s + self.eta_a * a)


def build_pair_model(
    arm: str,
    dictionary: np.ndarray,
    subspace: cssd.CommonSubspace,
    *,
    seed: int = SEED,
    scale_seed: int = SCALE_SEED,
    eta_s: float = 0.0,
    eta_a: float = 0.0,
    gamma: float = 1.0,
) -> Any:
    """Fresh Full model for one arm with the exact same parameter RNG stream."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known {ARMS}")
    clean = cssd.CSSD_SPEC
    common = dict(
        subspace=subspace,
        spec=sc.FULL,
        node_binding=clean.node_binding,
        edge_binding=clean.edge_binding,
        coding=clean.coding,
        scale_seed=int(scale_seed),
    )
    torch.manual_seed(int(seed))
    if arm == "residual":
        return ResidualNodeBindingScaleModel(
            cm.H1_CONFIG,
            dictionary,
            eta_s=float(eta_s),
            eta_a=float(eta_a),
            gamma=float(gamma),
            **common,
        )
    return sc.LatentScaleSEM108(cm.H1_CONFIG, dictionary, **common)


def set_residual_coefficients(model: Any, coeffs: "ResidualCoefficients") -> None:
    if not isinstance(model, ResidualNodeBindingScaleModel):
        raise TypeError("residual coefficients require the residual arm model")
    with torch.no_grad():
        model.eta_s.fill_(float(coeffs.eta_s))
        model.eta_a.fill_(float(coeffs.eta_a))
        model.gamma.fill_(float(coeffs.gamma))


def parameter_state_hash(model: torch.nn.Module) -> str:
    state = model.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# fixed-amplitude residual coefficients
# ---------------------------------------------------------------------------


@dataclass
class ResidualCoefficients:
    eta_s: float
    eta_a: float
    gamma: float
    rms_u0: float
    rms_s: float
    rms_a: float
    rms_slot0: float
    rms_slot1: float
    rms_slot1_scaled: float
    n_occurrences: int
    n_slot_entries: int
    graph_ids: list[int]
    graph_ids_sha256: str
    selection_seed: int
    n_graphs: int
    init_state_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "eta_s": self.eta_s,
            "eta_a": self.eta_a,
            "gamma": self.gamma,
            "rms_u0": self.rms_u0,
            "rms_s": self.rms_s,
            "rms_a": self.rms_a,
            "rms_slot0": self.rms_slot0,
            "rms_slot1_unscaled": self.rms_slot1,
            "rms_slot1_scaled": self.rms_slot1_scaled,
            "slot_rms_match_error": float(abs(self.rms_slot0 - self.rms_slot1_scaled)),
            "slot_rms_match_rel_error": float(
                abs(self.rms_slot0 - self.rms_slot1_scaled) / max(self.rms_slot0, 1e-12)
            ),
            "n_occurrences": self.n_occurrences,
            "n_slot_entries": self.n_slot_entries,
            "graph_ids": list(self.graph_ids),
            "graph_ids_sha256": self.graph_ids_sha256,
            "selection_seed": self.selection_seed,
            "n_graphs": self.n_graphs,
            "init_state_sha256": self.init_state_sha256,
            "official_test_loaded": False,
        }


def _binding_terms(model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coord = model.code(batch.dict_phi)
    q = F.one_hot(batch.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
    occ = batch.env_occ_node.to(coord.device)
    c = coord[occ]
    qc = q[occ]
    s = c @ model.W_A_S
    a = qc @ model.W_A_C
    u0 = (s * a) / math.sqrt(float(p2.D_A))
    index = batch.env_occ_root.to(coord.device) * int(p2.N_SHELLS) + batch.env_occ_shell.to(coord.device)
    return s, a, u0, index


def compute_residual_coefficients(
    model: torch.nn.Module,
    train_data: Sequence[Any],
    device: torch.device,
    *,
    n_graphs: int = COEFF_GRAPHS,
    selection_seed: int = SEED,
    batch_size: int = BATCH_SIZE,
) -> ResidualCoefficients:
    """Fixed coefficients from the canonical shared initial state, train only.

    Uses the uncentred RMS of (u0, s, a) over the occurrences of a fixed random
    train sample; ``gamma`` then matches the aggregated 3-shell slot RMS.
    """
    n_total = int(len(train_data))
    k = min(int(n_graphs), n_total)
    rng = np.random.default_rng(int(selection_seed))
    selected = np.sort(rng.choice(n_total, size=k, replace=False))
    subset = [train_data[int(i)] for i in selected]
    model = model.to(device).eval()
    loader = p1lib.make_env_loader(subset, int(batch_size), False, int(SEED) + int(uprun.EVAL_SHUFFLE_OFFSET))

    sq_s = sq_a = sq_u0 = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            s, a, u0, _index = _binding_terms(model, batch)
            sq_s += float((s * s).sum())
            sq_a += float((a * a).sum())
            sq_u0 += float((u0 * u0).sum())
            count += int(s.shape[0])
    rms_s = math.sqrt(sq_s / max(count, 1))
    rms_a = math.sqrt(sq_a / max(count, 1))
    rms_u0 = math.sqrt(sq_u0 / max(count, 1))
    if not all(math.isfinite(v) and v > 1e-12 for v in (rms_s, rms_a, rms_u0)):
        raise RuntimeError("INIT_INVALID: non-finite or degenerate initial binding RMS")
    eta_s = rms_u0 / rms_s
    eta_a = rms_u0 / rms_a
    if not all(math.isfinite(v) and v > 1e-12 for v in (eta_s, eta_a)):
        raise RuntimeError("INIT_INVALID: non-finite additive coefficients")

    sq_S0 = sq_S1 = 0.0
    n_slot_entries = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            s, a, u0, index = _binding_terms(model, batch)
            u1 = u0 + float(eta_s) * s + float(eta_a) * a
            n = int(batch.dict_phi.shape[0]) if hasattr(batch, "dict_phi") else int(batch.num_nodes)
            flat0 = torch.zeros(
                (n * int(p2.N_SHELLS), int(p2.D_A)), device=u0.device, dtype=u0.dtype
            )
            flat1 = torch.zeros_like(flat0)
            flat0.index_add_(0, index, u0)
            flat1.index_add_(0, index, u1)
            sq_S0 += float((flat0 * flat0).sum())
            sq_S1 += float((flat1 * flat1).sum())
            n_slot_entries += int(flat0.shape[0])
    rms_slot0 = math.sqrt(sq_S0 / max(n_slot_entries, 1))
    rms_slot1 = math.sqrt(sq_S1 / max(n_slot_entries, 1))
    if not all(math.isfinite(v) and v > 1e-12 for v in (rms_slot0, rms_slot1)):
        raise RuntimeError("INIT_INVALID: non-finite aggregated slot RMS")
    gamma = rms_slot0 / rms_slot1
    if not math.isfinite(gamma) or gamma <= 1e-12:
        raise RuntimeError("INIT_INVALID: non-finite gamma")

    ids_sha = hashlib.sha256(json.dumps([int(i) for i in selected]).encode()).hexdigest()
    return ResidualCoefficients(
        eta_s=float(eta_s),
        eta_a=float(eta_a),
        gamma=float(gamma),
        rms_u0=float(rms_u0),
        rms_s=float(rms_s),
        rms_a=float(rms_a),
        rms_slot0=float(rms_slot0),
        rms_slot1=float(rms_slot1),
        rms_slot1_scaled=float(gamma * rms_slot1),
        n_occurrences=int(count),
        n_slot_entries=int(n_slot_entries),
        graph_ids=[int(i) for i in selected],
        graph_ids_sha256=ids_sha,
        selection_seed=int(selection_seed),
        n_graphs=int(k),
        init_state_sha256=parameter_state_hash(model),
    )


# ---------------------------------------------------------------------------
# node-health diagnostics
# ---------------------------------------------------------------------------


def _param_stats(param: torch.Tensor, init: torch.Tensor | None) -> dict[str, Any]:
    value = param.detach()
    out = {
        "norm": float(value.norm().item()),
        "absmax": float(value.abs().max().item()) if value.numel() else 0.0,
    }
    if init is not None:
        denom = float(init.norm().item())
        out["rel_change"] = float((value - init).norm().item() / denom) if denom > 0 else 0.0
    else:
        out["rel_change"] = 0.0
    return out


def node_health(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    *,
    epoch: int,
    init_state: Mapping[str, torch.Tensor] | None,
    eta_s: float,
    eta_a: float,
    gamma: float,
) -> dict[str, Any]:
    """One fixed-batch snapshot.  No optimizer step; eval mode (no RNG use)."""
    batch = batch.to(device)
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    handle_pre = model.node_encoder.register_forward_pre_hook(
        lambda _m, inputs: captured.__setitem__("node_slots", inputs[0].detach())
    )
    handle_post = model.node_encoder.register_forward_hook(
        lambda _m, _inp, out: captured.__setitem__("node_out", out.detach())
    )
    with torch.no_grad():
        model(batch, mask=cm.C6_MASK)
    handle_pre.remove()
    handle_post.remove()
    node_slots = captured["node_slots"].reshape(captured["node_slots"].shape[0], -1)
    node_out = captured["node_out"].reshape(captured["node_out"].shape[0], -1)

    graph_index = batch.batch.to(node_slots.device)
    n_graphs = int(graph_index.max().item()) + 1 if int(graph_index.numel()) else 0

    def _per_graph_rms(values: torch.Tensor) -> tuple[float, float]:
        if n_graphs <= 0:
            return 0.0, 0.0
        sumsq = torch.zeros(n_graphs, device=values.device, dtype=values.dtype)
        count = torch.zeros(n_graphs, device=values.device, dtype=values.dtype)
        sumsq.index_add_(0, graph_index, (values * values).sum(dim=1))
        count.index_add_(0, graph_index, torch.ones_like(values[:, 0]))
        rms = torch.sqrt(sumsq / count.clamp_min(1.0))
        return float(rms.mean().item()), float(rms.std(unbiased=False).item())

    slot_rms_mean, slot_rms_std = _per_graph_rms(node_slots)
    out_rms_mean, out_rms_std = _per_graph_rms(node_out)

    with torch.no_grad():
        s, a, u0, _index = _binding_terms(model, batch)
    perm = torch.roll(torch.arange(a.shape[0], device=a.device), shifts=1)
    mixed = (s * a[perm]) / math.sqrt(float(p2.D_A))

    def _rms(x: torch.Tensor) -> float:
        return float(torch.sqrt((x * x).mean()).item()) if x.numel() else 0.0

    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "node_slot": {
            "rms": _rms(node_slots),
            "zero_frac": float((node_slots == 0).float().mean().item()) if node_slots.numel() else 1.0,
            "cross_sample_std": float(node_slots.std(dim=0).mean().item()) if node_slots.numel() else 0.0,
            "absmax": float(node_slots.abs().max().item()) if node_slots.numel() else 0.0,
            "per_graph_rms_mean": slot_rms_mean,
            "per_graph_rms_std": slot_rms_std,
        },
        "node_out": {
            "rms": _rms(node_out),
            "per_graph_rms_mean": out_rms_mean,
            "per_graph_rms_std": out_rms_std,
            "per_column_std_mean": float(node_out.std(dim=0).mean().item()),
            "constant_column_frac": float((node_out.std(dim=0) < 1e-8).float().mean().item()),
            "zero_column_frac": float((node_out.abs().max(dim=0).values < 1e-12).float().mean().item()),
        },
        "terms": {
            "product_rms": _rms(u0),
            "structural_additive_rms": _rms(float(eta_s) * s),
            "atomic_additive_rms": _rms(float(eta_a) * a),
            "cross_mix_delta_rms": _rms(u0 - mixed),
            "cross_term_zero_frac": float((u0 == 0).float().mean().item()) if u0.numel() else 1.0,
        },
        "params": {
            "W_A_S": _param_stats(model.W_A_S, None if init_state is None else init_state["W_A_S"]),
            "W_A_C": _param_stats(model.W_A_C, None if init_state is None else init_state["W_A_C"]),
            "node_encoder_first": _param_stats(
                model.node_encoder[0].weight,
                None if init_state is None else init_state["node_encoder_first"],
            ),
        },
    }

    # task-only gradient reachability (no step, gradients cleared afterwards)
    model.eval()
    prediction, _aux = model(batch, mask=cm.C6_MASK, return_aux=True)
    task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    model.zero_grad(set_to_none=True)
    task.backward()
    grad_payload: dict[str, Any] = {}
    for name, param in (
        ("W_A_S", model.W_A_S),
        ("W_A_C", model.W_A_C),
        ("node_encoder_first", model.node_encoder[0].weight),
    ):
        grad = param.grad
        if grad is None:
            grad_payload[name] = {"grad_norm": 0.0, "grad_zero_frac": 1.0}
        else:
            grad_payload[name] = {
                "grad_norm": float(grad.norm().item()),
                "grad_zero_frac": float((grad == 0).float().mean().item()),
            }
    model.zero_grad(set_to_none=True)
    payload["task_grad"] = grad_payload
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _soup_state(store: Mapping[int, Mapping[str, torch.Tensor]], members: Sequence[int]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([store[epoch][key].float() for epoch in members]).mean(0)
        for key in store[members[0]]
    }


def train_arm(
    arm: str,
    model: torch.nn.Module,
    train_data: Sequence[Any],
    valid_data: Sequence[Any] | None,
    *,
    device: torch.device,
    out_dir: Path,
    epochs: int = EPOCHS,
    lr: float = LR,
    coeffs: ResidualCoefficients | None = None,
    diag_epochs: Sequence[int] = DIAG_EPOCHS,
) -> dict[str, Any]:
    """Fresh Full trajectory; loss / optimizer / masking follow the parent recipe."""
    seed_everything(SEED)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=WEIGHT_DECAY)
    loader = p1lib.make_env_loader(train_data, BATCH_SIZE, True, SEED + int(uprun.TRAIN_SHUFFLE_OFFSET))
    eval_loader = (
        p1lib.make_env_loader(valid_data, BATCH_SIZE, False, SEED + int(uprun.EVAL_SHUFFLE_OFFSET))
        if valid_data is not None
        else None
    )
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)

    diag_loader = p1lib.make_env_loader(
        train_data, DIAG_BATCH_SIZE, False, SEED + int(uprun.EVAL_SHUFFLE_OFFSET)
    )
    diag_batch = next(iter(diag_loader))
    init_state = {
        "W_A_S": model.W_A_S.detach().clone(),
        "W_A_C": model.W_A_C.detach().clone(),
        "node_encoder_first": model.node_encoder[0].weight.detach().clone(),
    }
    eta_s = float(getattr(model, "eta_s", 0.0))
    eta_a = float(getattr(model, "eta_a", 0.0))
    gamma = float(getattr(model, "gamma", 1.0))

    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    epoch_seconds: list[float] = []
    peak_memory = 0.0
    started = time.perf_counter()

    health.append(
        node_health(
            model, diag_batch, device, epoch=0, init_state=init_state,
            eta_s=eta_s, eta_a=eta_a, gamma=gamma,
        )
    )
    print(
        f"[{arm}] epoch=000 init slot_rms={health[0]['node_slot']['rms']:.6e} "
        f"zero_frac={health[0]['node_slot']['zero_frac']:.3f} "
        f"node_out_const={health[0]['node_out']['constant_column_frac']:.3f}",
        flush=True,
    )

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum = 0.0
        n_molecules = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_molecules += int(batch.y.numel())

        valid_mae = float("nan")
        if eval_loader is not None:
            model.eval()
            with torch.no_grad():
                predictions: list[torch.Tensor] = []
                targets: list[torch.Tensor] = []
                for batch in eval_loader:
                    batch = batch.to(device)
                    predictions.append(model(batch, mask=mask).view(-1).detach().cpu())
                    targets.append(batch.y.view(-1).detach().cpu())
            prediction = torch.cat(predictions)
            target = torch.cat(targets)
            valid_mae = float((prediction - target).abs().mean())
        train_mae = float(task_sum / max(n_molecules, 1))
        epoch_seconds.append(float(time.perf_counter() - epoch_started))
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "valid_mae": valid_mae,
                "seconds": epoch_seconds[-1],
            }
        )
        if int(epoch) in set(int(e) for e in SOUP_EPOCHS):
            soup[int(epoch)] = {
                key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()
            }
        if device.type == "cuda":
            peak_memory = max(peak_memory, float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)))
        if int(epoch) in set(int(e) for e in diag_epochs):
            health.append(
                node_health(
                    model, diag_batch, device, epoch=int(epoch), init_state=init_state,
                    eta_s=eta_s, eta_a=eta_a, gamma=gamma,
                )
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            h = health[-1]
            print(
                f"[{arm}] epoch={epoch:03d} train={train_mae:.6f} valid={valid_mae:.6f} "
                f"{epoch_seconds[-1]:.2f}s slot_rms={h['node_slot']['rms']:.4e} "
                f"slot_zero={h['node_slot']['zero_frac']:.3f} "
                f"grad_WA_S={h['task_grad']['W_A_S']['grad_norm']:.3e}",
                flush=True,
            )

    members = sorted(soup)
    soup_state = _soup_state(soup, members) if members else {}
    result: dict[str, Any] = {
        "arm": arm,
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(WEIGHT_DECAY),
        "batch_size": int(BATCH_SIZE),
        "grad_clip": float(GRAD_CLIP),
        "soup_epochs": list(SOUP_EPOCHS),
        "members": members,
        "curve": curve,
        "health": health,
        "wall_clock_s": float(time.perf_counter() - started),
        "seconds_per_epoch": float(sum(epoch_seconds) / max(len(epoch_seconds), 1)),
        "peak_gpu_memory_mb": peak_memory,
        "soup_state": soup_state,
        "official_test_loaded": False,
    }
    if coeffs is not None:
        result["coefficients"] = coeffs.as_dict()

    # publication predictions on the saved soup
    if valid_data is not None and members:
        fresh = build_pair_model(
            arm,
            uprun._dictionary_tensor(),
            uprun._load_parent_subspace(),
            eta_s=(coeffs.eta_s if coeffs is not None else 0.0),
            eta_a=(coeffs.eta_a if coeffs is not None else 0.0),
            gamma=(coeffs.gamma if coeffs is not None else 1.0),
        )
        fresh.to(device)
        fresh.load_state_dict(soup_state)
        raw_valid, y_valid = _collect(fresh, valid_data, device)
        raw_train, y_train = _collect(fresh, train_data, device)
        bias = float(np.median(y_train - raw_train))
        cal_valid = raw_valid + bias
        cal_train = raw_train + bias
        seen = set()
        curve_by_epoch = {int(row["epoch"]): row for row in curve}
        member_valid = [float(curve_by_epoch[epoch]["valid_mae"]) for epoch in members if epoch in curve_by_epoch]
        result.update(
            {
                "raw_valid_mae": float(np.mean(np.abs(raw_valid - y_valid))),
                "calibrated_valid_mae": float(np.mean(np.abs(cal_valid - y_valid))),
                "raw_train_mae": float(np.mean(np.abs(raw_train - y_train))),
                "calibrated_train_mae": float(np.mean(np.abs(cal_train - y_train))),
                "train_fitted_bias": bias,
                "eval_gap": float(np.mean(np.abs(cal_valid - y_valid)) - np.mean(np.abs(cal_train - y_train))),
                "member_valid_mae": member_valid,
                "valid_predictions_raw": raw_valid.tolist(),
                "valid_predictions_calibrated": cal_valid.tolist(),
                "valid_targets": y_valid.tolist(),
                "raw_valid_mae_eval": float(np.mean(np.abs(raw_valid - y_valid))),
            }
        )
    return result


def _collect(model: torch.nn.Module, data: Sequence[Any], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = p1lib.make_env_loader(data, BATCH_SIZE, False, SEED + int(uprun.EVAL_SHUFFLE_OFFSET))
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
            targets.append(batch.y.view(-1).detach().cpu())
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()


# ---------------------------------------------------------------------------
# correctness checks (CPU-friendly)
# ---------------------------------------------------------------------------


def run_checks(n_graphs: int = 32, device: torch.device | None = None) -> dict[str, Any]:
    device = torch.device("cpu") if device is None else device
    seed_everything(SEED)
    subspace = uprun._load_parent_subspace()
    dictionary = uprun._dictionary_tensor()
    train_data = uprun.load_split("control", "train")
    subset = list(train_data[: int(n_graphs)])
    loader = p1lib.make_env_loader(subset, int(n_graphs), False, SEED + int(uprun.EVAL_SHUFFLE_OFFSET))
    batch = next(iter(loader)).to(device)

    control = build_pair_model("control", dictionary, subspace).to(device).eval()
    residual_zero = build_pair_model("residual", dictionary, subspace, eta_s=0.0, eta_a=0.0, gamma=1.0).to(device).eval()
    residual = build_pair_model("residual", dictionary, subspace, eta_s=1.0, eta_a=1.0, gamma=1.0).to(device).eval()

    checks: dict[str, Any] = {"official_test_loaded": False, "n_graphs": int(n_graphs)}
    checks["parameter_state_hash"] = {
        "control": parameter_state_hash(control),
        "residual_zero": parameter_state_hash(residual_zero),
    }
    checks["parameter_state_identical"] = bool(
        checks["parameter_state_hash"]["control"] == checks["parameter_state_hash"]["residual_zero"]
    )
    checks["trainable_params"] = {
        "control": int(sum(p.numel() for p in control.parameters() if p.requires_grad)),
        "residual": int(sum(p.numel() for p in residual.parameters() if p.requires_grad)),
    }
    checks["trainable_params_identical"] = bool(
        checks["trainable_params"]["control"] == checks["trainable_params"]["residual"]
    )

    with torch.no_grad():
        coord = control.code(batch.dict_phi)
        q = F.one_hot(batch.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        occ = batch.env_occ_node.to(coord.device)
        c = coord[occ]
        qc = q[occ]
        reference = (c @ control.W_A_S) * (qc @ control.W_A_C) / math.sqrt(float(p2.D_A))
        hook_value = control._node_binding(c, qc)
        checks["product_hook_max_abs_diff"] = float((reference - hook_value).abs().max().item())
        checks["product_hook_equivalent"] = bool(checks["product_hook_max_abs_diff"] == 0.0)

        z = residual_zero._node_binding(c, qc)
        checks["residual_zero_reduction_max_abs_diff"] = float((z - reference).abs().max().item())
        checks["residual_zero_reduces_to_product"] = bool(
            checks["residual_zero_reduction_max_abs_diff"] <= 1e-12
        )
        r = residual._node_binding(c, qc)
        checks["residual_differs_from_product_max_abs"] = float((r - reference).abs().max().item())

    # forward uses the hook and node_binding_zero zeroes the whole u (additive included)
    def _capture_slots(m: torch.nn.Module, mask: Any) -> torch.Tensor:
        store: dict[str, torch.Tensor] = {}
        handle = m.node_encoder.register_forward_pre_hook(
            lambda _mod, inputs: store.__setitem__("slots", inputs[0].detach())
        )
        with torch.no_grad():
            m(batch, mask=mask)
        handle.remove()
        return store["slots"].clone()

    control_slots = _capture_slots(control, cm.C6_MASK)
    residual_slots = _capture_slots(residual, cm.C6_MASK)
    zero_slots = _capture_slots(residual, _zero_mask())
    checks["control_slot_rms"] = float(torch.sqrt((control_slots ** 2).mean()).item())
    checks["residual_slot_rms"] = float(torch.sqrt((residual_slots ** 2).mean()).item())
    checks["residual_slot_changes"] = bool(
        float((residual_slots - control_slots).abs().max().item()) > 1e-9
    )
    checks["zero_mask_slot_absmax"] = float(zero_slots.abs().max().item())
    checks["zero_mask_zeroes_all"] = bool(checks["zero_mask_slot_absmax"] == 0.0)

    # task-only gradient reaches the three binding/encoder groups on the residual arm
    residual.train()
    prediction, aux = residual(batch, mask=cm.C6_MASK, return_aux=True)
    task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    residual.zero_grad(set_to_none=True)
    task.backward()
    grad_norms = {
        name: float(param.grad.norm().item()) if param.grad is not None else 0.0
        for name, param in (
            ("W_A_S", residual.W_A_S),
            ("W_A_C", residual.W_A_C),
            ("node_encoder_first", residual.node_encoder[0].weight),
        )
    }
    residual.zero_grad(set_to_none=True)
    checks["task_grad_norms"] = grad_norms
    checks["task_grad_reaches_all"] = bool(all(value > 1e-12 for value in grad_norms.values()))
    checks["all_passed"] = bool(
        checks["parameter_state_identical"]
        and checks["trainable_params_identical"]
        and checks["product_hook_equivalent"]
        and checks["residual_zero_reduces_to_product"]
        and checks["residual_slot_changes"]
        and checks["zero_mask_zeroes_all"]
        and checks["task_grad_reaches_all"]
    )
    return checks


def _zero_mask() -> Any:
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit

    return audit.AuditMask(node_binding_zero=True)


# ---------------------------------------------------------------------------
# export helpers
# ---------------------------------------------------------------------------


def write_curve(out_dir: Path, curve: Sequence[Mapping[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mae", "valid_mae", "seconds"])
        writer.writeheader()
        for row in curve:
            writer.writerow(row)


def write_valid_predictions(out_dir: Path, result: Mapping[str, Any]) -> None:
    raw = result["valid_predictions_raw"]
    cal = result["valid_predictions_calibrated"]
    y = result["valid_targets"]
    bias = result["train_fitted_bias"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "valid_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["graph_id", "y", "pred_raw", "train_fitted_bias", "pred_calibrated", "abs_error"])
        for index, (yt, pr, pc) in enumerate(zip(y, raw, cal)):
            writer.writerow([index, yt, pr, bias, pc, abs(pc - yt)])


__all__ = [
    "PROTOCOL_VERSION",
    "EXPORT_DIR",
    "ARMS",
    "SEED",
    "SCALE_SEED",
    "EPOCHS",
    "LR",
    "WEIGHT_DECAY",
    "GRAD_CLIP",
    "BATCH_SIZE",
    "SOUP_EPOCHS",
    "DIAG_EPOCHS",
    "ResidualNodeBindingScaleModel",
    "ResidualCoefficients",
    "seed_everything",
    "build_pair_model",
    "set_residual_coefficients",
    "parameter_state_hash",
    "compute_residual_coefficients",
    "node_health",
    "train_arm",
    "run_checks",
    "write_curve",
    "write_valid_predictions",
]