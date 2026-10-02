"""ZINC overnight bottleneck v1 (2026-10-02/03).

Two independent, pre-registered directions from the canonical fresh Full
initialisation (shared structure dictionary + structure/atom-semantics binding +
local-environment statistics; no message passing / transformer):

* **Node-collapse 2x2** (arms ``N0``..``N3``).  Factor A = initial node-slot
  amplitude (``kappa = 1`` vs ``kappa = 1 / r_init``, a fixed non-trainable
  buffer multiplying the *same* product binding).  Factor B = weight decay on
  the node parameters only (``W_A_S``, ``W_A_C``, ``node_encoder.*``) at the
  original ``1e-5`` vs ``0``.  ``N0`` is the shared control.
* **Additive topology readout** (arm ``T0``).  The graph reader keeps
  ``GenericReader_other(R[:, :806])`` and the original topology block gets an
  independent ``Linear(8, 1)`` scalar head; both branches are summed.  The
  node path is the *original* product binding / WD / amplitude.

Every arm starts from the same canonical fresh initialisation, uses the
original optimizer / LR / WD / loss / clip / batch / epoch budget, and is
selected only by the frozen rules in the preregistration.  The official ZINC
**test** split is never instantiated.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1lib
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_node_binding_residual_pair_v1 as rp
from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as uprun
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader

PROTOCOL_VERSION = "zinc-overnight-bottleneck-v1"
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
EXPORT_DIR = TRACK_ROOT / "results/zinc_overnight_bottleneck_v1"
FROZEN_SCALE_PATH = EXPORT_DIR / "frozen_node_scale_seed0.json"

SEED = 0
SCALE_SEED = 0
EPOCHS = 240
PREFIX_EPOCHS = 80
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
DIAG_EPOCHS = (0, 1, 10, 20, 40, 80, 160, 240)
SCALE_GRAPHS = 1024
DIAG_BATCH_SIZE = 128
NODE_WD_PREFIXES = ("W_A_S", "W_A_C", "node_encoder.")
G0_CONTRIB_WORSENING_MAX = 0.001
NODE_GAIN_MIN = 0.003
T_GAIN_WITHOUT172_MIN = -0.0005
WEAK_TASK_USE_RESPONSE = 1.0e-4

NODE_ARMS = ("N0", "N1", "N2", "N3")
ARMS = ("N0", "N1", "N2", "N3", "T0")
#: the two fixed, diverse train sentinel batches (eval order, no shuffle).
SENTINEL_A = (0, 128)
SENTINEL_B = (2048, 2176)

_write_json = uprun._write_json


# ---------------------------------------------------------------------------
# seeding / device
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rng_state_cuda() -> Any:
    if torch.cuda.is_available():
        return [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return None


def _restore_cuda_rng(states: Any) -> None:
    if states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.to("cuda") for state in states])


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class UnitScaleNodeBindingModel(sc.LatentScaleSEM108):
    """Full model with a fixed initial-RMS unit scale on the product binding.

    ``kappa`` is a non-trainable buffer; by linearity of the shell aggregation
    this is exactly a uniform scaling of every node slot, so the aggregated
    slot RMS at initialisation becomes 1.  No new shape, parameter, gate or
    activation.  ``kappa = 1`` is bit-identical to the frozen product binding.
    """

    def __init__(self, config: Any, dictionary: np.ndarray, *, subspace: cssd.CommonSubspace, kappa: float = 1.0, **kwargs: Any) -> None:
        super().__init__(config, dictionary, subspace=subspace, **kwargs)
        self.register_buffer("kappa", torch.tensor(float(kappa)), persistent=False)
        self.binding_mode = "unit_scale"

    def _node_binding(self, c: torch.Tensor, qc: torch.Tensor) -> torch.Tensor:
        s = c @ self.W_A_S
        a = qc @ self.W_A_C
        return self.kappa * (s * a) / math.sqrt(float(p2.D_A))


class AdditiveReader(nn.Module):
    """``GenericReader_other(R[:, :other_dim]) + Linear(topo_dim, 1)(R[:, other_dim:])``.

    Identical hidden layers and the first ``other_dim`` input columns of the
    canonical ``GenericReader``; the topology columns are moved out of the
    shared first layer into an independent scalar head, so the explicit
    topology contribution is no longer gated by the other coordinates.
    """

    STYLE = "additive_topology"

    def __init__(self, reader: GenericReader, other_dim: int, *, head_seed: int = 0) -> None:
        super().__init__()
        net = reader.net
        first = net[0]
        self.other_dim = int(other_dim)
        self.topo_dim = int(first.weight.shape[1]) - int(other_dim)
        if self.topo_dim <= 0:
            raise ValueError("other_dim must be smaller than the reader input width")
        self.other = nn.Linear(self.other_dim, int(first.weight.shape[0]))
        with torch.no_grad():
            self.other.weight.copy_(first.weight[:, : self.other_dim])
            self.other.bias.copy_(first.bias)
        self.tail = nn.Sequential(*list(net.children())[1:])
        self.topo = nn.Linear(self.topo_dim, 1)
        generator = torch.Generator().manual_seed(int(head_seed))
        bound = 1.0 / math.sqrt(float(self.topo_dim))
        with torch.no_grad():
            self.topo.weight.uniform_(-bound, bound, generator=generator)
            self.topo.bias.uniform_(-bound, bound, generator=generator)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        other = self.tail(self.other(z[:, : self.other_dim]))
        topo = self.topo(z[:, self.other_dim :])
        return (other + topo).squeeze(-1)


def build_readout_model(
    dictionary: np.ndarray,
    subspace: cssd.CommonSubspace,
    *,
    seed: int = SEED,
    scale_seed: int = SCALE_SEED,
    additive_topology: bool = False,
    head_seed: int = 0,
) -> Any:
    """Canonical fresh Full; optionally replace the reader with AdditiveReader."""
    torch.manual_seed(int(seed))
    model = sc.LatentScaleSEM108(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        spec=sc.FULL,
        node_binding=cssd.CSSD_SPEC.node_binding,
        edge_binding=cssd.CSSD_SPEC.edge_binding,
        coding=cssd.CSSD_SPEC.coding,
        scale_seed=int(scale_seed),
    )
    if additive_topology:
        topo_out = int(p2.TOPOLOGY_OUT)
        reader_in = int(sc.FULL.reader_input)
        model.reader = AdditiveReader(model.reader, reader_in - topo_out, head_seed=int(head_seed))
        model.readout_style = AdditiveReader.STYLE
    else:
        model.readout_style = "shared_reader"
    return model


def build_node_model(
    arm: str,
    dictionary: np.ndarray,
    subspace: cssd.CommonSubspace,
    *,
    kappa: float,
    seed: int = SEED,
    scale_seed: int = SCALE_SEED,
) -> Any:
    if arm in ("N1", "N3"):
        torch.manual_seed(int(seed))
        model = UnitScaleNodeBindingModel(
            cm.H1_CONFIG,
            dictionary,
            subspace=subspace,
            kappa=float(kappa),
            spec=sc.FULL,
            node_binding=cssd.CSSD_SPEC.node_binding,
            edge_binding=cssd.CSSD_SPEC.edge_binding,
            coding=cssd.CSSD_SPEC.coding,
            scale_seed=int(scale_seed),
        )
        model.readout_style = "shared_reader"
        return model
    return build_readout_model(dictionary, subspace, seed=seed, scale_seed=scale_seed)


def parameter_state_hash(model: torch.nn.Module) -> str:
    state = model.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# frozen initial node scale (factor A)
# ---------------------------------------------------------------------------


@dataclass
class NodeScaleManifest:
    kappa: float
    r_init_slot_rms: float
    rms_u0: float
    n_graphs: int
    graph_ids_sha256: str
    init_state_sha256: str
    selection_seed: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kappa": self.kappa,
            "r_init_slot_rms": self.r_init_slot_rms,
            "rms_u0": self.rms_u0,
            "n_graphs": self.n_graphs,
            "graph_ids_sha256": self.graph_ids_sha256,
            "init_state_sha256": self.init_state_sha256,
            "selection_seed": self.selection_seed,
            "seed": self.seed,
            "rule": "kappa = 1 / per-entry uncentred RMS of the aggregated 3-shell product slots on a fixed seed-0 1024-graph train sample",
            "official_test_loaded": False,
        }


def compute_node_scale(model: torch.nn.Module, train_data: Sequence[Any], device: torch.device, *, n_graphs: int = SCALE_GRAPHS, seed: int = SEED) -> NodeScaleManifest:
    """``kappa = 1 / r_init`` where ``r_init`` is the per-entry uncentred RMS of
    the aggregated 3-shell product slots over a fixed seed-0 train sample.

    Per-entry (not per-slot-vector) is the frozen convention: the expected
    magnitude of the canonical init is ~2.4e-4, and scaling by ``kappa`` makes
    the aggregated slot RMS exactly 1 in the same units the node health probe
    reports (``node_slot.rms``).
    """
    n_total = int(len(train_data))
    k = min(int(n_graphs), n_total)
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(n_total, size=k, replace=False))
    subset = [train_data[int(i)] for i in selected]
    model = model.to(device).eval()
    loader = p1lib.make_env_loader(subset, BATCH_SIZE, False, SEED + int(uprun.EVAL_SHUFFLE_OFFSET))
    sumsq = 0.0
    n_entries = 0
    rms_u0_sq = 0.0
    n_occ = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            coord = model.code(batch.dict_phi)
            q = F.one_hot(batch.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
            occ = batch.env_occ_node.to(device)
            c = coord[occ]
            qc = q[occ]
            u0 = (c @ model.W_A_S) * (qc @ model.W_A_C) / math.sqrt(float(p2.D_A))
            n = int(batch.dict_phi.shape[0])
            flat = torch.zeros((n * int(p2.N_SHELLS), int(p2.D_A)), device=u0.device, dtype=u0.dtype)
            index = batch.env_occ_root.to(device) * int(p2.N_SHELLS) + batch.env_occ_shell.to(device)
            flat.index_add_(0, index, u0)
            sumsq += float((flat.double() ** 2).sum())
            n_entries += int(flat.numel())
            rms_u0_sq += float((u0.double() ** 2).sum())
            n_occ += int(u0.shape[0])
    r_init = math.sqrt(sumsq / max(n_entries, 1))
    rms_u0 = math.sqrt(rms_u0_sq / max(n_occ, 1))
    if not math.isfinite(r_init) or r_init <= 1e-12:
        raise RuntimeError(f"INIT_INVALID: initial aggregated slot RMS {r_init!r}")
    ids_sha = hashlib.sha256(json.dumps([int(i) for i in selected]).encode()).hexdigest()
    return NodeScaleManifest(
        kappa=float(1.0 / r_init),
        r_init_slot_rms=r_init,
        rms_u0=rms_u0,
        n_graphs=int(k),
        graph_ids_sha256=ids_sha,
        init_state_sha256=parameter_state_hash(model),
        selection_seed=int(seed),
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# optimizer (node-WD factor)
# ---------------------------------------------------------------------------


def build_optimizer(model: torch.nn.Module, *, lr: float, node_weight_decay: float) -> torch.optim.Adam:
    """Original Adam with exactly two groups; every parameter appears once.

    Group ``other`` keeps the original ``1e-5``; group ``node`` carries the
    frozen node weight decay (``1e-5`` for N0/N1, ``0`` for N2/N3).  With
    ``node_weight_decay == WEIGHT_DECAY`` this is mathematically the original
    single-group Adam.
    """
    seen: set[int] = set()
    node_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if id(param) in seen:
            raise RuntimeError(f"duplicate parameter object in named_parameters: {name}")
        seen.add(id(param))
        (node_params if name.startswith(NODE_WD_PREFIXES) else other_params).append(param)
    if not node_params:
        raise RuntimeError("no node weight-decay parameters matched")
    return torch.optim.Adam(
        [
            {"params": other_params, "weight_decay": float(WEIGHT_DECAY), "group": "other"},
            {"params": node_params, "weight_decay": float(node_weight_decay), "group": "node"},
        ],
        lr=float(lr),
    )


def parameter_group_inventory(model: torch.nn.Module) -> dict[str, list[dict[str, Any]]]:
    node, other = [], []
    for name, param in model.named_parameters():
        entry = {"name": name, "shape": list(param.shape), "numel": int(param.numel())}
        (node if name.startswith(NODE_WD_PREFIXES) else other).append(entry)
    return {"node": node, "other": other}


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt((x.reshape(-1).to(torch.float64) ** 2).mean()).item()) if x.numel() else 0.0


def node_probe(model: torch.nn.Module, batch: Any, device: torch.device, *, epoch: int, init_max_col_std: float | None = None) -> dict[str, Any]:
    """Fixed-batch node snapshot.  eval mode, no optimizer step, no RNG use."""
    batch = batch.to(device)
    was_training = model.training
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    h_pre = model.node_encoder.register_forward_pre_hook(lambda _m, inp: captured.__setitem__("slots", inp[0].detach()))
    h_post = model.node_encoder.register_forward_hook(lambda _m, _i, out: captured.__setitem__("out", out.detach()))
    with torch.no_grad():
        model(batch, mask=cm.C6_MASK)
    h_pre.remove()
    h_post.remove()
    slots = captured["slots"].reshape(captured["slots"].shape[0], -1)
    out = captured["out"].reshape(captured["out"].shape[0], -1)

    graph_index = batch.batch.to(slots.device)
    n_graphs = int(graph_index.max().item()) + 1 if int(graph_index.numel()) else 0

    def per_graph_rms(values: torch.Tensor) -> tuple[float, float]:
        if n_graphs <= 0:
            return 0.0, 0.0
        sumsq = torch.zeros(n_graphs, device=values.device, dtype=values.dtype)
        count = torch.zeros(n_graphs, device=values.device, dtype=values.dtype)
        sumsq.index_add_(0, graph_index, (values * values).sum(dim=1))
        count.index_add_(0, graph_index, torch.ones_like(values[:, 0]))
        rms = torch.sqrt(sumsq / count.clamp_min(1.0))
        return float(rms.mean().item()), float(rms.std(unbiased=False).item())

    slot_rms_mean, slot_rms_std = per_graph_rms(slots)
    out_rms_mean, out_rms_std = per_graph_rms(out)
    max_col_std = float(out.std(dim=0).max().item())
    threshold = None
    if init_max_col_std is not None:
        threshold = max(1e-8, 1e-3 * float(init_max_col_std))

    # binding terms
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        q = F.one_hot(batch.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        occ = batch.env_occ_node.to(coord.device)
        c = coord[occ]
        qc = q[occ]
        s = c @ model.W_A_S
        a = qc @ model.W_A_C
        u0 = (s * a) / math.sqrt(float(p2.D_A))

    # task-only gradient reachability (no step; eval mode so dropout never consumes RNG)
    prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
    task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    model.zero_grad(set_to_none=True)
    task.backward()
    grad: dict[str, dict[str, float]] = {}
    for name, param in (("W_A_S", model.W_A_S), ("W_A_C", model.W_A_C), ("node_encoder_first", model.node_encoder[0].weight)):
        g = param.grad
        grad[name] = {"grad_norm": float(g.norm().item()) if g is not None else 0.0, "finite": bool(torch.isfinite(g).all().item()) if g is not None else False}
    model.zero_grad(set_to_none=True)

    # dependency response: replace node_out with its own batch mean
    def _replace_node_out(_m: Any, _i: Any, o: torch.Tensor) -> torch.Tensor:
        return o.mean(dim=0, keepdim=True).expand_as(o)

    with torch.no_grad():
        base = model(batch, mask=cm.C6_MASK)
        handle = model.node_encoder.register_forward_hook(_replace_node_out)
        swapped = model(batch, mask=cm.C6_MASK)
        handle.remove()
    dep_delta = _rms(swapped - base)
    dep_rel = dep_delta / max(_rms(base), 1e-12)
    model.train(was_training)

    payload = {
        "epoch": int(epoch),
        "node_slot": {
            "rms": _rms(slots),
            "zero_frac": float((slots == 0).float().mean().item()) if slots.numel() else 1.0,
            "absmax": float(slots.abs().max().item()) if slots.numel() else 0.0,
            "per_graph_rms_mean": slot_rms_mean,
            "per_graph_rms_std": slot_rms_std,
        },
        "node_out": {
            "rms": _rms(out),
            "per_graph_rms_mean": out_rms_mean,
            "per_graph_rms_std": out_rms_std,
            "max_col_std": max_col_std,
            "alive_threshold": threshold,
            "alive": bool(threshold is not None and max_col_std > threshold),
        },
        "terms": {
            "product_rms": _rms(u0),
            "product_zero_frac": float((u0 == 0).float().mean().item()) if u0.numel() else 1.0,
        },
        "task_grad": grad,
        "dependency": {"delta_rms": dep_delta, "relative": dep_rel, "weak_task_use": bool(dep_delta <= WEAK_TASK_USE_RESPONSE)},
    }
    return payload


def sentinel_batches(train_data: Sequence[Any]) -> list[Any]:
    n = len(train_data)
    if n < 2 * DIAG_BATCH_SIZE:
        raise ValueError(f"need at least {2 * DIAG_BATCH_SIZE} train graphs for two sentinel batches, got {n}")
    a = list(train_data[SENTINEL_A[0] : SENTINEL_A[1]])
    b_start = SENTINEL_B[0] if n >= SENTINEL_B[1] else max(0, n - DIAG_BATCH_SIZE)
    b = list(train_data[b_start : b_start + DIAG_BATCH_SIZE])
    loader_a = p1lib.make_env_loader(a, len(a), False, SEED + int(uprun.EVAL_SHUFFLE_OFFSET))
    loader_b = p1lib.make_env_loader(b, len(b), False, SEED + int(uprun.EVAL_SHUFFLE_OFFSET))
    return [next(iter(loader_a)), next(iter(loader_b))]


def health_gate(probes: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen numeric purchase threshold (not a proof of task value)."""
    per_batch = {}
    ok = True
    for tag, probe in probes.items():
        g = probe["task_grad"]
        grads_finite = all(g[name]["finite"] for name in g)
        grad_alive = any(g[name]["grad_norm"] > 1e-10 for name in g)
        product_nonzero = probe["terms"]["product_rms"] > 0.0
        alive = bool(probe["node_out"]["alive"])
        healthy = bool(alive and grads_finite and grad_alive and product_nonzero)
        ok = ok and healthy
        per_batch[tag] = {
            "node_out_alive": alive,
            "max_col_std": probe["node_out"]["max_col_std"],
            "threshold": probe["node_out"]["alive_threshold"],
            "grads_finite": grads_finite,
            "grad_alive": grad_alive,
            "product_nonzero": product_nonzero,
            "healthy": healthy,
            "weak_task_use": probe["dependency"]["weak_task_use"],
        }
    return {"healthy": ok, "per_batch": per_batch}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _soup_state(store: Mapping[int, Mapping[str, torch.Tensor]], members: Sequence[int]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([store[int(e)][key].float() for e in members]).mean(0) for key in store[int(members[0])]}


def _export_curve(out_dir: Path, curve: Sequence[Mapping[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mae", "valid_mae", "seconds"])
        writer.writeheader()
        for row in curve:
            writer.writerow({k: row[k] for k in ("epoch", "train_mae", "valid_mae", "seconds")})


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
    node_weight_decay: float = WEIGHT_DECAY,
    kappa: float = 1.0,
    diag_epochs: Sequence[int] = DIAG_EPOCHS,
    resume_state: Mapping[str, Any] | None = None,
    checkpoint_path: Path | None = None,
    train_seed: int = SEED,
) -> dict[str, Any]:
    """One fresh (or resumed) Full trajectory; original loss / optimizer recipe."""
    seed_everything(int(train_seed))
    model.to(device)
    optimizer = build_optimizer(model, lr=lr, node_weight_decay=node_weight_decay)
    generator = torch.Generator().manual_seed(int(train_seed) + int(uprun.TRAIN_SHUFFLE_OFFSET))
    loader = DataLoader(
        list(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=p1lib.env_collate,
    )
    eval_loader = (
        p1lib.make_env_loader(valid_data, BATCH_SIZE, False, int(train_seed) + int(uprun.EVAL_SHUFFLE_OFFSET))
        if valid_data is not None
        else None
    )
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)
    sentinels = sentinel_batches(train_data)
    init_max_col_std = [probe_init_max_col_std(model, batch, device) for batch in sentinels]

    start_epoch = 0
    curve: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    soup: dict[int, dict[str, torch.Tensor]] = {}
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        generator.set_state(resume_state["generator"])
        torch.set_rng_state(resume_state["torch_rng"])
        _restore_cuda_rng(resume_state.get("cuda_rng"))
        np.random.set_state(resume_state["numpy_rng"])
        random.setstate(resume_state["python_rng"])
        start_epoch = int(resume_state["epoch"])
        curve = list(resume_state.get("curve", []))
        health = list(resume_state.get("health", []))
        print(f"[{arm}] resumed at epoch {start_epoch}", flush=True)

    init_state = {
        "W_A_S": model.W_A_S.detach().clone(),
        "W_A_C": model.W_A_C.detach().clone(),
        "node_encoder_first": model.node_encoder[0].weight.detach().clone(),
    }
    epoch_seconds: list[float] = []
    peak_memory = 0.0
    started = time.perf_counter()

    if start_epoch == 0:
        health.append(probe_pair(arm, model, sentinels, device, epoch=0, init_max_col_std=init_max_col_std))
        print(f"[{arm}] epoch=000 slot_rms={health[0]['batch_a']['node_slot']['rms']:.4e} node_out_alive={health[0]['batch_a']['node_out']['alive']}", flush=True)

    for epoch in range(start_epoch + 1, int(epochs) + 1):
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
                preds, targets = [], []
                for batch in eval_loader:
                    batch = batch.to(device)
                    preds.append(model(batch, mask=mask).view(-1).detach().cpu())
                    targets.append(batch.y.view(-1).detach().cpu())
                valid_mae = float((torch.cat(preds) - torch.cat(targets)).abs().mean())
        epoch_seconds.append(float(time.perf_counter() - epoch_started))
        curve.append({"epoch": int(epoch), "train_mae": float(task_sum / max(n_molecules, 1)), "valid_mae": valid_mae, "seconds": epoch_seconds[-1]})
        if int(epoch) in set(int(e) for e in SOUP_EPOCHS):
            soup[int(epoch)] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if device.type == "cuda":
            peak_memory = max(peak_memory, float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)))
        if int(epoch) in set(int(e) for e in diag_epochs):
            health.append(probe_pair(arm, model, sentinels, device, epoch=int(epoch), init_max_col_std=init_max_col_std))
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            h = health[-1]["batch_a"]
            print(
                f"[{arm}] epoch={epoch:03d} train={curve[-1]['train_mae']:.6f} valid={valid_mae:.6f} "
                f"{epoch_seconds[-1]:.2f}s slot_rms={h['node_slot']['rms']:.4e} node_out_std={h['node_out']['max_col_std']:.3e} "
                f"grad_WA_S={h['task_grad']['W_A_S']['grad_norm']:.3e}",
                flush=True,
            )
        if checkpoint_path is not None and (int(epoch) == PREFIX_EPOCHS or int(epoch) == int(epochs) or int(epoch) == 160):
            save_checkpoint(checkpoint_path, arm=arm, model=model, optimizer=optimizer, generator=generator, epoch=int(epoch), curve=curve, health=health, kappa=kappa, train_seed=train_seed)

    members = sorted(soup)
    soup_state = _soup_state(soup, members) if members else {}
    result: dict[str, Any] = {
        "arm": arm,
        "epochs": int(epochs),
        "resumed_from": int(start_epoch),
        "lr": float(lr),
        "weight_decay": float(WEIGHT_DECAY),
        "node_weight_decay": float(node_weight_decay),
        "batch_size": int(BATCH_SIZE),
        "grad_clip": float(GRAD_CLIP),
        "train_seed": int(train_seed),
        "kappa": float(kappa),
        "soup_epochs": list(SOUP_EPOCHS),
        "members": members,
        "curve": curve,
        "health": health,
        "wall_clock_s": float(time.perf_counter() - started),
        "seconds_per_epoch": float(sum(epoch_seconds) / max(len(epoch_seconds), 1)),
        "peak_gpu_memory_mb": peak_memory,
        "soup_state": soup_state,
        "init_max_col_std": [float(v) for v in init_max_col_std],
        "official_test_loaded": False,
    }
    if valid_data is not None and members:
        fresh = _rebuild_for_eval(arm, model, kappa)
        fresh.to(device)
        fresh.load_state_dict(soup_state)
        raw_valid, y_valid = _collect(fresh, valid_data, device)
        raw_train, y_train = _collect(fresh, train_data, device)
        bias = float(np.median(y_train - raw_train))
        cal_valid = raw_valid + bias
        cal_train = raw_train + bias
        curve_by_epoch = {int(row["epoch"]): row for row in curve}
        result.update(
            {
                "raw_valid_mae": float(np.mean(np.abs(raw_valid - y_valid))),
                "calibrated_valid_mae": float(np.mean(np.abs(cal_valid - y_valid))),
                "raw_train_mae": float(np.mean(np.abs(raw_train - y_train))),
                "calibrated_train_mae": float(np.mean(np.abs(cal_train - y_train))),
                "train_fitted_bias": bias,
                "eval_gap": float(np.mean(np.abs(cal_valid - y_valid)) - np.mean(np.abs(cal_train - y_train))),
                "member_valid_mae": [float(curve_by_epoch[e]["valid_mae"]) for e in members if e in curve_by_epoch],
                "valid_predictions_raw": raw_valid.tolist(),
                "valid_predictions_calibrated": cal_valid.tolist(),
                "valid_targets": y_valid.tolist(),
            }
        )
    return result


def _rebuild_for_eval(arm: str, model: torch.nn.Module, kappa: float) -> torch.nn.Module:
    """Rebuild the same architecture for publication evaluation (no init RNG)."""
    dictionary = uprun._dictionary_tensor()
    subspace = uprun._load_parent_subspace()
    if arm == "T0":
        return build_readout_model(dictionary, subspace, additive_topology=True)
    return build_node_model(arm, dictionary, subspace, kappa=kappa)


def probe_init_max_col_std(model: torch.nn.Module, batch: Any, device: torch.device) -> float:
    batch = batch.to(device)
    was_training = model.training
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    handle = model.node_encoder.register_forward_hook(lambda _m, _i, out: captured.__setitem__("out", out.detach()))
    with torch.no_grad():
        model(batch, mask=cm.C6_MASK)
    handle.remove()
    out = captured["out"].reshape(captured["out"].shape[0], -1)
    model.train(was_training)
    return float(out.std(dim=0).max().item())


def probe_pair(arm: str, model: torch.nn.Module, sentinels: Sequence[Any], device: torch.device, *, epoch: int, init_max_col_std: Sequence[float]) -> dict[str, Any]:
    probes = {
        "batch_a": node_probe(model, sentinels[0], device, epoch=epoch, init_max_col_std=init_max_col_std[0]),
        "batch_b": node_probe(model, sentinels[1], device, epoch=epoch, init_max_col_std=init_max_col_std[1]),
    }
    return {"arm": arm, "epoch": int(epoch), **probes, "gate": health_gate(probes)}


def save_checkpoint(path: Path, *, arm: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, generator: torch.Generator, epoch: int, curve: Sequence[Mapping[str, Any]], health: Sequence[Mapping[str, Any]], kappa: float, train_seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arm": arm,
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": _rng_state_cuda(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "curve": list(curve),
            "health": list(health),
            "kappa": float(kappa),
            "train_seed": int(train_seed),
            "official_test_loaded": False,
        },
        path,
    )


def _collect(model: torch.nn.Module, data: Sequence[Any], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = p1lib.make_env_loader(data, BATCH_SIZE, False, SEED + int(uprun.EVAL_SHUFFLE_OFFSET))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
            targets.append(batch.y.view(-1).detach().cpu())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


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


# ---------------------------------------------------------------------------
# correctness checks
# ---------------------------------------------------------------------------


def _zero_mask() -> Any:
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit

    return audit.AuditMask(node_binding_zero=True)


def run_checks(n_graphs: int = 32) -> dict[str, Any]:
    device = torch.device("cpu")
    seed_everything(SEED)
    subspace = uprun._load_parent_subspace()
    dictionary = uprun._dictionary_tensor()
    train = uprun.load_split("control", "train")
    subset = list(train[: int(n_graphs)])
    batch = next(iter(p1lib.make_env_loader(subset, int(n_graphs), False, SEED + int(uprun.EVAL_SHUFFLE_OFFSET)))).to(device)

    checks: dict[str, Any] = {"official_test_loaded": False, "n_graphs": int(n_graphs)}
    n0 = build_readout_model(dictionary, subspace).to(device).eval()
    t0 = build_readout_model(dictionary, subspace, additive_topology=True).to(device).eval()
    n1 = build_node_model("N1", dictionary, subspace, kappa=2.0).to(device).eval()
    n0_hash = parameter_state_hash(n0)

    checks["node_arms_share_init"] = {}
    for arm in ("N1", "N2", "N3"):
        model = build_node_model(arm, dictionary, subspace, kappa=2.0 if arm in ("N1", "N3") else 1.0)
        checks["node_arms_share_init"][arm] = bool(parameter_state_hash(model) == n0_hash)
        del model
    checks["node_arms_share_init_all"] = bool(all(checks["node_arms_share_init"].values()))

    n1_unit = build_node_model("N1", dictionary, subspace, kappa=1.0).to(device).eval()
    with torch.no_grad():
        coord = n0.code(batch.dict_phi)
        q = F.one_hot(batch.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        occ = batch.env_occ_node.to(device)
        c, qc = coord[occ], q[occ]
        reference = n1_unit._node_binding(c, qc)
        checks["kappa1_reduces_to_product_max_abs_diff"] = float((reference - (c @ n0.W_A_S) * (qc @ n0.W_A_C) / math.sqrt(float(p2.D_A))).abs().max().item())
        checks["kappa1_reduces_to_product"] = bool(checks["kappa1_reduces_to_product_max_abs_diff"] == 0.0)
        checks["kappa2_scales_by_2_max_abs_diff"] = float((2.0 * reference - n1._node_binding(c, qc)).abs().max().item())

        rms_n0 = _slot_rms(n0, batch)
        rms_n1 = _slot_rms(n1, batch)
        checks["n0_slot_rms"] = rms_n0
        checks["n1_slot_rms"] = rms_n1
        checks["n1_slot_rms_over_n0_slot_rms"] = float(rms_n1 / max(rms_n0, 1e-12))
        checks["kappa2_slot_ratio_is_2"] = bool(abs(checks["n1_slot_rms_over_n0_slot_rms"] - 2.0) < 1e-4)

    # AdditiveReader structural + gradient checks
    reader_in = int(sc.FULL.reader_input)
    other_dim = reader_in - int(p2.TOPOLOGY_OUT)
    z = torch.randn(4, reader_in, device=device)
    z_other = z.clone()
    z_other[:, other_dim:] = 0.0
    with torch.no_grad():
        canonical_other_branch = n0.reader(z_other)
        additive_other_branch = t0.reader.tail(t0.reader.other(z[:, :other_dim])).squeeze(-1)
    checks["additive_branch_equivalence_to_shared_columns_max_abs_diff"] = float((canonical_other_branch - additive_other_branch).abs().max().item())
    checks["additive_head_nonzero"] = bool(float(t0.reader.topo.weight.abs().sum().item()) > 0.0)
    checks["additive_uses_shared_tail"] = bool(t0.reader.tail is not None)
    # gradients reach other branch, topology head, and the topology encoder
    t0.train()
    t0.zero_grad(set_to_none=True)
    pred, aux = t0(batch, mask=cm.C6_MASK, return_aux=True)
    F.l1_loss(pred.view(-1), batch.y.view(-1)).backward()
    grad_norms = {
        "other.weight": float(t0.reader.other.weight.grad.norm().item()),
        "tail_first.weight": float(t0.reader.tail[1].weight.grad.norm().item()),
        "topo.weight": float(t0.reader.topo.weight.grad.norm().item()),
        "topology_encoder_first.weight": float(t0.topology_encoder[0].weight.grad.norm().item()),
    }
    t0.zero_grad(set_to_none=True)
    checks["additive_grad_norms"] = grad_norms
    checks["additive_grad_reaches_all"] = bool(all(v > 1e-12 for v in grad_norms.values()))

    # node_binding_zero zeroes the whole binding for the scaled arm
    checks["zero_mask_slot_absmax"] = _zero_mask_slot_absmax(n1, batch)
    checks["zero_mask_zeroes_all"] = bool(checks["zero_mask_slot_absmax"] == 0.0)

    checks["parameter_counts"] = {
        "N0": int(sum(p.numel() for p in n0.parameters() if p.requires_grad)),
        "T0": int(sum(p.numel() for p in t0.parameters() if p.requires_grad)),
    }
    checks["all_passed"] = bool(
        checks["node_arms_share_init_all"]
        and checks["kappa1_reduces_to_product"]
        and checks["kappa2_slot_ratio_is_2"]
        and checks["additive_branch_equivalence_to_shared_columns_max_abs_diff"] < 1e-9
        and checks["additive_head_nonzero"]
        and checks["additive_grad_reaches_all"]
        and checks["zero_mask_zeroes_all"]
    )
    return checks


def _capture_slots(model: torch.nn.Module, batch: Any, *, zero: bool = False) -> torch.Tensor:
    store: dict[str, torch.Tensor] = {}
    handle = model.node_encoder.register_forward_pre_hook(lambda _m, inputs: store.__setitem__("slots", inputs[0].detach()))
    mask = _zero_mask() if zero else cm.C6_MASK
    with torch.no_grad():
        model(batch, mask=mask)
    handle.remove()
    return store["slots"]


def _slot_rms(model: torch.nn.Module, batch: Any) -> float:
    return _rms(_capture_slots(model, batch))


def _zero_mask_slot_absmax(model: torch.nn.Module, batch: Any) -> float:
    return float(_capture_slots(model, batch, zero=True).abs().max().item())


__all__ = [
    "PROTOCOL_VERSION",
    "EXPORT_DIR",
    "FROZEN_SCALE_PATH",
    "ARMS",
    "NODE_ARMS",
    "EPOCHS",
    "PREFIX_EPOCHS",
    "SOUP_EPOCHS",
    "DIAG_EPOCHS",
    "SCALE_GRAPHS",
    "NODE_WD_PREFIXES",
    "UnitScaleNodeBindingModel",
    "AdditiveReader",
    "NodeScaleManifest",
    "build_readout_model",
    "build_node_model",
    "build_optimizer",
    "parameter_group_inventory",
    "parameter_state_hash",
    "compute_node_scale",
    "node_probe",
    "sentinel_batches",
    "health_gate",
    "train_arm",
    "save_checkpoint",
    "write_valid_predictions",
    "run_checks",
]