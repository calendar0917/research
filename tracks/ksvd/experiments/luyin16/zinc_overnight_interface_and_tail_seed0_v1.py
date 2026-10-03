"""ZINC overnight interface + tail, seed 0 — main (GPU) runner.

Round ``zinc_overnight_interface_and_tail_seed0_v1``.  Two questions:

1. **Main path.**  On top of the *compressed* ``S_M`` reference ``B``
   (``zinc_zero_binding_baseline_seed0_v1``, 267,611 params, fusion input
   ``[Sem108; size2]``), does a *direct* residual interface that reads
   ``struct_code (x) atom/bond category`` joint statistics buy generalisation?
   The interface is inserted **after the fusion output** (width 144) and
   **before the shared task dictionary bridge**:

       h = h0 + delta(F),  delta = Linear(5102, 24) -> SiLU -> Linear(24, 144)

   with the last layer zero-initialised, so at initialisation the model is
   ``B`` exactly.  ``F = [Sem110, K_A, K_E]``; ``K_A``/``K_E`` are the node /
   edge joint tensors built from the *original* ``S_M`` structural code
   (65 -> common coordinate + tied-IHT / dense residual code, ``D_star``, ``U``
   from the ``S_M`` raw soup).  The interface is a *linear readout of the joint
   tensor*; it is not the old multiplicative slot path and not a graph-level
   head.

   Phase 1 (this file, ``--mode phase1``): parent frozen and eval, only the
   adapter trains.  Five arms ``R_CS/R_SM/R_SJ/R_DM/R_DJ`` differ only in the
   residual input block (Sem only / sparse marginal / sparse joint / dense
   marginal / dense joint).  Zero-init means step-1 adapter gradients *can* be
   zero; the first-layer gradients become live from the first update.

   Phase 2 (``--mode phase2``): parent, ``D`` and adapter all train (matched
   arms ``A0/C/T``), loss ``L1(y) + H1_LAMBDA * reconstruction``.

   Phase 3 (``--mode phase3``): the same matched pair retrained on all 10,000
   official-train rows from the same compressed 8k start (no dev holdout).

2. **Tail (CPU, separate runner).**  Balanced-severity cycle-head weighting.

Train-only: ``encoded_train.pt`` + ``env_train.pt``.  Official-valid is not
read here; official-test is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_structure_semantic_factorial_seed0_v1 as zsf
from tracks.ksvd.experiments.luyin16 import zinc_zero_binding_baseline_seed0_v1 as zbn

PROTOCOL_VERSION = "zinc-overnight-interface-and-tail-seed0-v1"

TRACK_ROOT = zjd.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/zinc_overnight_interface_and_tail_seed0_v1"

SOURCE_FACTORIAL = TRACK_ROOT / "results/zinc_structure_semantic_factorial_seed0_v1"
SOURCE_ZERO_BINDING = TRACK_ROOT / "results/zinc_zero_binding_baseline_seed0_v1"
S_M_RAW_SOUP = SOURCE_FACTORIAL / "S_M_raw_soup_state.pt"
S_M_SOUP_SHA256 = "2def4cb64c1350a60348fb6fc7ce88a07bffe563367ac7e9666ee6f7dff5c5b8"
DEPLOY_STATE = SOURCE_ZERO_BINDING / "S_M_deploy_state.pt"
PHASE_A_REPLAY = SOURCE_ZERO_BINDING / "phaseA_sm_replay.npz"
SOURCE_BRANCH = "task/zinc-structure-semantic-factorial-seed0-v1"
SOURCE_BRANCH_HEAD = "4083ee38d8785c5a44d1f9628a5baeed568d4c34"
SOURCE_TRAIN_COMMIT = "118e2481362f"
ZERO_BINDING_EXEC_COMMIT = "dc96fac"
ZERO_BINDING_RESULT_COMMIT = "53ab987"

SEED = 0
SCALE_SEED = 0
ADAPTER_SEED = 0
TRAIN_SHUFFLE_OFFSET = 101
BASE_EPOCHS = 240
PHASE1_EPOCHS = 160
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
H1_LAMBDA = float(cm.H1_LAMBDA)
COMPRESS_TOL = 1.0e-5
BOOT_SEED = 20261003
N_BOOT = 1000
DELTA = 0.003
GROUP_KEYS = ("k0", "k-1", "kle-2")
GROUP_NAMES = {"k0": "k=0", "k-1": "k=-1", "kle-2": "k<=-2"}

DEPLOY_PARAMETERS = 267_611
D_PARAMETERS = 65 * 32
ADAPTER_IN = 110 + 3 * 32 * 28 + 6 * 96 * 4  # 5102
ADAPTER_HIDDEN = 24
FUSION_OUT = 144
ADAPTER_PARAMETERS = ADAPTER_IN * ADAPTER_HIDDEN + ADAPTER_HIDDEN + ADAPTER_HIDDEN * FUSION_OUT + FUSION_OUT

N_SHELLS = int(p2.N_SHELLS)
SHELLPAIR_CLASSES = int(p2.SHELLPAIR_CLASSES)
ATOM_CATEGORIES = int(p2.ATOM_CATEGORIES)
BOND_CATEGORIES = int(p2.BOND_CATEGORIES)
K_ATOMS = int(cssd.K_ATOMS)
NODE_BLOCK = N_SHELLS * K_ATOMS * ATOM_CATEGORIES  # 2688
EDGE_BLOCK = SHELLPAIR_CLASSES * 3 * K_ATOMS * BOND_CATEGORIES  # 2304

# arm -> (code mode, block mode); block "none" zeroes both structural blocks.
ARM_SPEC: dict[str, tuple[str, str]] = {
    "R_CS": ("none", "none"),
    "R_SM": ("sparse", "marginal"),
    "R_SJ": ("sparse", "joint"),
    "R_DM": ("dense", "marginal"),
    "R_DJ": ("dense", "joint"),
}
PHASE1_ARMS = tuple(ARM_SPEC)
PHASE2_ARMS = ("A0", "C", "T")
TAU_KEYS = ("tau_A", "tau_E")


# ---------------------------------------------------------------------------
# helpers (source conventions)
# ---------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def schedule_hash(n: int, epochs: int, offset: int, batch_size: int = BATCH_SIZE) -> dict[str, str]:
    gen = torch.Generator().manual_seed(int(SEED) + int(offset))
    whole = hashlib.sha256()
    first = None
    for _ in range(int(epochs)):
        for indices in zftd.epoch_batches(int(n), batch_size, gen, True):
            arr = np.asarray(indices, np.int64)
            whole.update(arr.tobytes())
            if first is None:
                first = hashlib.sha256(arr.tobytes()).hexdigest()
    return {"full_schedule_sha256": whole.hexdigest(), "first_batch_sha256": first}


def resolve_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key in ("cuda", "cuda:0", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    raise ValueError(f"unsupported device {name!r}")


# ---------------------------------------------------------------------------
# data (train-only)
# ---------------------------------------------------------------------------


def load_data() -> tuple[dict[str, Any], dict[str, np.ndarray], list[Any], list[Any], list[Any], np.ndarray, np.ndarray, np.ndarray]:
    blob = zsf.load_prep_blob()
    decomp = zsf.load_target()
    train_data = zsf.load_train_only()
    zftd.apply_prep_train_only(train_data, blob)
    fit_data, dev_data = zsf.build_fit_dev(train_data, blob["fit_idx"], blob["dev_idx"])
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    y = np.asarray(decomp["y"], np.float64)
    return blob, decomp, train_data, fit_data, dev_data, fit_idx, dev_idx, y


def load_dictionary_objects() -> dict[str, Any]:
    """``D_star``/``U``/``common_rms``/``kappa`` from the trained S_M raw soup."""
    state = torch.load(S_M_RAW_SOUP, map_location="cpu", weights_only=False)
    got = zsf.state_hash(state)
    if got != S_M_SOUP_SHA256:
        raise RuntimeError(f"S_M raw soup hash mismatch: {got}")
    return {
        "D": state["D"].clone(),
        "U": state["U"].clone(),
        "common_rms": state["common_rms"].clone(),
        "kappa": float(state["kappa"].reshape(-1)[0].item()),
        "soup_state_sha256": got,
    }


def load_deploy_state() -> dict[str, torch.Tensor]:
    state = torch.load(DEPLOY_STATE, map_location="cpu", weights_only=False)
    if int(sum(v.numel() for v in state.values() if torch.is_tensor(v))) != DEPLOY_PARAMETERS:
        raise RuntimeError("compressed deploy state parameter count mismatch")
    return state


# ---------------------------------------------------------------------------
# interface operators
# ---------------------------------------------------------------------------


def joint_node_block(alpha: torch.Tensor, data: Any, block: str) -> torch.Tensor:
    """``[n*N_SHELLS, 32*28]`` node joint (``joint``) or marginal (``marginal``)."""
    n = int(alpha.shape[0])
    dev = alpha.device
    index = data.env_occ_root.to(dev) * N_SHELLS + data.env_occ_shell.to(dev)
    node = data.env_occ_node.to(dev)
    a_v = alpha[node]
    t_v = F.one_hot(data.dict_atom.to(dev)[node], num_classes=ATOM_CATEGORIES).to(alpha.dtype)
    if block == "joint":
        outer = (a_v.unsqueeze(2) * t_v.unsqueeze(1)).reshape(int(a_v.shape[0]), K_ATOMS * ATOM_CATEGORIES)
        out = alpha.new_zeros((n * N_SHELLS, K_ATOMS * ATOM_CATEGORIES))
        out.index_add_(0, index, outer)
        return out
    if block == "marginal":
        sum_a = alpha.new_zeros((n * N_SHELLS, K_ATOMS))
        sum_t = alpha.new_zeros((n * N_SHELLS, ATOM_CATEGORIES))
        count = alpha.new_zeros((n * N_SHELLS,))
        sum_a.index_add_(0, index, a_v)
        sum_t.index_add_(0, index, t_v)
        count.index_add_(0, index, torch.ones_like(index, dtype=alpha.dtype))
        outer = sum_a.unsqueeze(2) * sum_t.unsqueeze(1)
        out = (outer / count.clamp_min(1.0).view(-1, 1, 1)).reshape(
            n * N_SHELLS, K_ATOMS * ATOM_CATEGORIES
        )
        return out.masked_fill((count <= 0).view(-1, 1), 0.0)
    raise ValueError(block)


def joint_edge_block(alpha: torch.Tensor, data: Any, block: str) -> torch.Tensor:
    """``[n*SHELLPAIR_CLASSES, 96*4]`` edge joint (``joint``) or marginal."""
    n = int(alpha.shape[0])
    dev = alpha.device
    index = data.env_bond_root.to(dev) * SHELLPAIR_CLASSES + data.env_bond_shellpair.to(dev)
    u = data.env_bond_u.to(dev)
    v = data.env_bond_v.to(dev)
    a_u = alpha[u]
    a_v = alpha[v]
    g = torch.cat([a_u + a_v, torch.abs(a_u - a_v), a_u * a_v], dim=1)
    b = F.one_hot(data.env_bond_type.to(dev), num_classes=BOND_CATEGORIES).to(alpha.dtype)
    width = 3 * K_ATOMS * BOND_CATEGORIES
    if block == "joint":
        outer = (g.unsqueeze(2) * b.unsqueeze(1)).reshape(int(g.shape[0]), width)
        out = alpha.new_zeros((n * SHELLPAIR_CLASSES, width))
        out.index_add_(0, index, outer)
        return out
    if block == "marginal":
        sum_g = alpha.new_zeros((n * SHELLPAIR_CLASSES, 3 * K_ATOMS))
        sum_b = alpha.new_zeros((n * SHELLPAIR_CLASSES, BOND_CATEGORIES))
        count = alpha.new_zeros((n * SHELLPAIR_CLASSES,))
        sum_g.index_add_(0, index, g)
        sum_b.index_add_(0, index, b)
        count.index_add_(0, index, torch.ones_like(index, dtype=alpha.dtype))
        outer = sum_g.unsqueeze(2) * sum_b.unsqueeze(1)
        out = (outer / count.clamp_min(1.0).view(-1, 1, 1)).reshape(
            n * SHELLPAIR_CLASSES, width
        )
        return out.masked_fill((count <= 0).view(-1, 1), 0.0)
    raise ValueError(block)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class InterfaceDeploy(zbn.DeployFull):
    """Compressed ``S_M`` body + zero-initialised direct interface adapter.

    The parent body is the exact compressed model; the structural dictionary is
    re-attached as ``self.D`` (a parameter so Phase 2/3 can train it) together
    with the frozen ``U``/``common_rms``/``kappa``.  ``code`` reconstructs the
    real structural code (the compressed placeholder is replaced); the adapter
    output is added to the *fusion output* before the shared task dictionary
    bridge.
    """

    def __init__(
        self,
        D: torch.Tensor,
        U: torch.Tensor,
        common_rms: torch.Tensor,
        kappa: float,
        tau_A: float,
        tau_E: float,
        *,
        code_mode: str = "sparse",
        block_mode: str = "joint",
    ) -> None:
        super().__init__()
        if code_mode not in ("sparse", "dense", "none"):
            raise ValueError(code_mode)
        if block_mode not in ("joint", "marginal", "none"):
            raise ValueError(block_mode)
        self.D = nn.Parameter(torch.as_tensor(D, dtype=torch.float32).clone())
        self.register_buffer("U", torch.as_tensor(U, dtype=torch.float32).clone())
        self.register_buffer("common_rms", torch.as_tensor(common_rms, dtype=torch.float32).clone())
        self.register_buffer("kappa", torch.tensor(float(kappa), dtype=torch.float32))
        self.register_buffer("tau_A", torch.tensor(float(tau_A), dtype=torch.float32))
        self.register_buffer("tau_E", torch.tensor(float(tau_E), dtype=torch.float32))
        self.code_mode = str(code_mode)
        self.block_mode = str(block_mode)
        self.adapter_zero = False
        # Adapter: private seed so every arm shares the same init regardless of
        # the parent builder's RNG stream.
        rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(int(ADAPTER_SEED))
            self.adapter = nn.Sequential(
                nn.Linear(ADAPTER_IN, ADAPTER_HIDDEN),
                nn.SiLU(),
                nn.Linear(ADAPTER_HIDDEN, FUSION_OUT),
            )
        finally:
            torch.set_rng_state(rng_state)
        with torch.no_grad():
            self.adapter[2].weight.zero_()
            self.adapter[2].bias.zero_()

    # -- coding ---------------------------------------------------------------

    def residual_dictionary(self) -> torch.Tensor:
        U = self.U.to(dtype=self.D.dtype)
        D_perp = self.D - U @ (U.t() @ self.D)
        return v0.normalized_dictionary(D_perp)

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        U = self.U.to(dtype=phi.dtype)
        c = phi @ U
        r = phi - c @ U.t()
        if self.code_mode == "sparse":
            alpha = v0.tied_iht_codes(
                self.residual_dictionary(), r, s=cssd.SPARSITY, steps=cssd.IHT_STEPS
            )
        elif self.code_mode == "dense":
            alpha = self.kappa.to(phi.dtype) * (r @ self.residual_dictionary())
        else:
            alpha = torch.zeros((int(phi.shape[0]), K_ATOMS), dtype=phi.dtype, device=phi.device)
        scale = self.common_rms.to(dtype=phi.dtype)
        return torch.cat([c / scale, alpha], dim=1)

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        if self.code_mode == "dense":
            alpha = coord[:, self.common_dim :] / self.kappa.to(coord.dtype)
            return alpha @ self.residual_dictionary().t()
        return super().reconstruct(phi, coord)

    # -- adapter input ---------------------------------------------------------

    def adapter_input(self, interface: torch.Tensor, coord: torch.Tensor, data: Any) -> torch.Tensor:
        alpha = coord[:, self.common_dim :]
        n = int(alpha.shape[0])
        if self.block_mode == "none":
            node = alpha.new_zeros((n, NODE_BLOCK))
            edge = alpha.new_zeros((n, EDGE_BLOCK))
        else:
            node = joint_node_block(alpha, data, self.block_mode).reshape(n, NODE_BLOCK) / self.tau_A.to(alpha.dtype)
            edge = joint_edge_block(alpha, data, self.block_mode).reshape(n, EDGE_BLOCK) / self.tau_E.to(alpha.dtype)
        return torch.cat([interface, node, edge], dim=1)

    # -- forward insertion point ----------------------------------------------

    def _environment_from_adapter(self, coord: torch.Tensor, data: Any, interface: torch.Tensor) -> torch.Tensor:
        h0 = self.fusion(interface)
        delta = self.adapter(self.adapter_input(interface, coord, data))
        if self.adapter_zero:
            delta = delta * 0.0
        return self.local_dictionary_bridge(h0 + delta)

    def environments_masked(self, coord, data, mask, fill=None):
        interface = sem.SEM108Model.semantic_interface(self, coord, data, mask, fill)
        return self._environment_from_adapter(coord, data, interface)

    def environments(self, coord, data):
        interface = sem.SEM108Model.semantic_interface(self, coord, data, None, None)
        return self._environment_from_adapter(coord, data, interface)


def build_interface_model(
    structural: Mapping[str, Any],
    stats: Mapping[str, float],
    *,
    code_mode: str,
    block_mode: str,
    device: torch.device | None = None,
    deploy_state: Mapping[str, torch.Tensor] | None = None,
) -> InterfaceDeploy:
    model = InterfaceDeploy(
        structural["D"],
        structural["U"],
        structural["common_rms"],
        float(stats.get("kappa", structural["kappa"])),
        float(stats["tau_A"]),
        float(stats["tau_E"]),
        code_mode=code_mode,
        block_mode=block_mode,
    )
    body = load_deploy_state() if deploy_state is None else deploy_state
    model.load_state_dict({k: v.to(torch.device("cpu")) for k, v in body.items()}, strict=False)
    if device is not None:
        model = model.to(device)
    return model


def interface_parameter_audit(model: InterfaceDeploy) -> dict[str, int]:
    body = 0
    d = int(model.D.numel())
    adapter = 0
    for name, parameter in model.named_parameters():
        if name == "D":
            d = int(parameter.numel())
        elif name.startswith("adapter."):
            adapter += int(parameter.numel())
        else:
            body += int(parameter.numel())
    return {
        "body_parameters": body,
        "D_parameters": d,
        "adapter_parameters": adapter,
        "total_parameters": body + d + adapter,
        "expected_adapter": ADAPTER_PARAMETERS,
    }


# ---------------------------------------------------------------------------
# interface statistics (tau / kappa) — computed once on all 8000 fit rows
# ---------------------------------------------------------------------------


def compute_interface_stats(
    model: InterfaceDeploy,
    fit_data: Sequence[Any],
    *,
    device: torch.device,
    chunk: int = 32,
    log: Any = print,
) -> dict[str, Any]:
    """``tau_A/tau_E`` from the Sparse+Joint blocks and ``kappa`` for Dense.

    All sums are float64 over every fit environment row and every element of
    the block.  ``kappa`` matches the dense-code global RMS to the sparse alpha
    RMS, exactly like ``zinc_structure_semantic_factorial_seed0_v1``.
    """
    t0 = time.perf_counter()
    Dbar = model.residual_dictionary().detach()
    U = model.U.detach()
    alpha_sq = dense_sq = node_sq = edge_sq = 0.0
    n_rows = 0
    with torch.no_grad():
        for start in range(0, len(fit_data), int(chunk)):
            chunk_data = fit_data[start : start + int(chunk)]
            batch = zftd.make_batch(
                chunk_data,
                list(range(len(chunk_data))),
                torch.zeros(len(chunk_data)),
                device,
            )
            phi = torch.cat(
                [d.dict_phi.to(dtype=torch.float32) for d in chunk_data], 0
            ).to(device)
            c = phi @ U
            r = phi - c @ U.t()
            alpha = v0.tied_iht_codes(Dbar, r, s=cssd.SPARSITY, steps=cssd.IHT_STEPS)
            dense = r @ Dbar
            alpha_sq += float((alpha.double() ** 2).sum())
            dense_sq += float((dense.double() ** 2).sum())
            node = joint_node_block(alpha, batch, "joint")
            edge = joint_edge_block(alpha, batch, "joint")
            node_sq += float((node.double() ** 2).sum())
            edge_sq += float((edge.double() ** 2).sum())
            n_rows += int(alpha.shape[0])
    if not (alpha_sq > 0 and dense_sq > 0 and node_sq > 0 and edge_sq > 0):
        raise RuntimeError("degenerate interface statistics")
    stats = {
        "tau_A": float(math.sqrt(node_sq / max(n_rows * NODE_BLOCK, 1))),
        "tau_E": float(math.sqrt(edge_sq / max(n_rows * EDGE_BLOCK, 1))),
        "kappa": float(math.sqrt(alpha_sq / dense_sq)),
        "sparse_alpha_rms": float(math.sqrt(alpha_sq / n_rows)),
        "dense_unscaled_rms": float(math.sqrt(dense_sq / n_rows)),
        "node_sq_sum": node_sq,
        "edge_sq_sum": edge_sq,
        "n_fit_env_rows": int(n_rows),
        "node_elements_per_row": NODE_BLOCK,
        "edge_elements_per_row": EDGE_BLOCK,
        "source_D": file_sha256(S_M_RAW_SOUP),
        "source_D_note": "D_star/U from S_M_raw_soup_state.pt (soup), D frozen in Phase 1",
        "seconds": float(time.perf_counter() - t0),
        "float64_accumulation": True,
        "denominator_clamp_min": 1.0e-8,
    }
    log(
        f"[stats] tau_A={stats['tau_A']:.6g} tau_E={stats['tau_E']:.6g} "
        f"kappa={stats['kappa']:.6g} rows={n_rows} {stats['seconds']:.1f}s"
    )
    return stats


def load_interface_stats(path: Path = RESULTS_DIR / "interface_stats.json") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_arm(
    arm: str,
    *,
    phase: int,
    code_mode: str,
    block_mode: str,
    device: torch.device,
    out_dir: Path = RESULTS_DIR,
    epochs: int,
    max_steps: int | None = None,
    train_all: bool = False,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    blob, decomp, train_data, fit_data, dev_data, fit_idx, dev_idx, y = load_data()
    if train_all:
        fit_data = list(train_data)
    target_fit = torch.as_tensor(y[fit_idx] if not train_all else y, dtype=torch.float32)
    structural = load_dictionary_objects()
    stats = load_interface_stats()
    if phase == 1:
        stats = dict(stats)
    else:
        # Phase 2/3 use the frozen sparse+joint tau and the frozen kappa.
        stats = dict(stats)

    seed_everything(SEED)
    model = build_interface_model(
        structural, stats, code_mode=code_mode, block_mode=block_mode, device=device
    ).to(device)
    audit = interface_parameter_audit(model)
    if audit["total_parameters"] != DEPLOY_PARAMETERS + D_PARAMETERS + ADAPTER_PARAMETERS:
        raise RuntimeError(f"interface parameter audit failed: {audit}")
    model.adapter_zero = bool(arm == "A0")
    # freeze contract
    if phase == 1:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("adapter."))
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    init_hash = zsf.trainable_parameter_hash(model)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    parent_init_hash = _parent_state_hash(model)

    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(int(SEED) + TRAIN_SHUFFLE_OFFSET)
    soup_epochs = set(range(max(1, int(epochs) - 4), int(epochs) + 1))
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    steps_done = 0
    stream = hashlib.sha256()
    started = time.perf_counter()
    stopped_reason = "completed"
    n_fit = len(fit_data)

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        if phase == 1:
            model.eval()  # frozen parent stays in eval; adapter is the only trained module
        else:
            model.train()
        task_sum, rec_sum, n_mol, n_steps, gnorm_sum, clip_hits = 0.0, 0.0, 0, 0, 0.0, 0
        for indices in zftd.epoch_batches(n_fit, BATCH_SIZE, train_gen, True):
            stream.update(np.asarray(indices, np.int64).tobytes())
            batch = zftd.make_batch(fit_data, indices, target_fit, device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            if phase == 1:
                loss = task
                rec_value = 0.0
            else:
                rec = model.reconstruction_loss(aux["phi"], aux["coord"])
                loss = task + H1_LAMBDA * rec
                rec_value = float(rec.detach())
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += rec_value
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "train_rec": float(rec_sum / max(n_steps, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "clip_fraction": float(clip_hits / max(n_steps, 1)),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if epoch in soup_epochs:
            soup[int(epoch)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[{arm}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"rec={curve[-1]['train_rec']:.4g} gnorm={curve[-1]['grad_norm']:.3g} "
                f"clip={curve[-1]['clip_fraction']:.3f} {curve[-1]['seconds']:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "phase": int(phase),
        "seed": SEED,
        "code_mode": code_mode,
        "block_mode": block_mode,
        "adapter_zero": bool(model.adapter_zero),
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "h1_lambda": H1_LAMBDA if phase != 1 else None,
        "soup_epochs": sorted(soup_epochs) if max_steps is None else [],
        "train_mode": "all10k" if train_all else "fit8k",
        "n_fit": int(n_fit),
        "parameter_audit": audit,
        "trainable_parameters": zsf.trainable_parameter_count(model),
        "init_trainable_hash": init_hash,
        "parent_init_state_hash": parent_init_hash,
        "tau": {"tau_A": float(model.tau_A), "tau_E": float(model.tau_E)},
        "kappa": float(model.kappa),
        "curve": curve,
        "data_stream_sha256": stream.hexdigest(),
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if max_steps is not None:
        result["last_trainable_hash"] = zsf.trainable_parameter_hash(model)
        result["parent_last_state_hash"] = _parent_state_hash(model)
        return result

    members = sorted(soup)
    soup_state = {k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]}
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(init_state, out_dir / f"{arm}_init_state.pt")

    replay = build_interface_model(
        structural, stats, code_mode=code_mode, block_mode=block_mode, device=device
    ).to(device)
    replay.load_state_dict({k: v.to(device) for k, v in soup_state.items()})
    replay.adapter_zero = bool(model.adapter_zero)
    replay.eval()
    fit_raw, _ = _predict(replay, fit_data, target_fit, device)
    y_fit_np = np.asarray(y if train_all else y[fit_idx], np.float64)
    b = float(np.median(y_fit_np - fit_raw))
    result.update(
        {
            "soup_members": members,
            "soup_state_sha256": zsf.state_hash(soup_state),
            "last_state_sha256": zsf.state_hash(last_state),
            "init_state_sha256": zsf.state_hash(init_state),
            "calibration": {"b": b, "folded_into_state": False, "computed_on": "fit8k" if not train_all else "train10k"},
            "fit_raw_mae_y": float(np.mean(np.abs(fit_raw - y_fit_np))),
            "fit_cal_mae_y": float(np.mean(np.abs(fit_raw + b - y_fit_np))),
        }
    )
    payload = {"fit_raw": fit_raw.astype(np.float32), "fit_idx": fit_idx if not train_all else np.arange(len(train_data)), "b": np.float64(b)}
    if not train_all:
        y_dev = np.asarray(y[dev_idx], np.float64)
        target_dev = torch.as_tensor(y_dev, dtype=torch.float32)
        dev_raw, _ = _predict(replay, dev_data, target_dev, device)
        payload.update({"dev_raw": dev_raw.astype(np.float32), "dev_idx": dev_idx})
        result["dev_raw_mae_y"] = float(np.mean(np.abs(dev_raw - y_dev)))
        k_dev = np.asarray(decomp["k"], np.int64)[dev_idx]
        g0 = k_dev == 0
        result["dev_g0_raw_mae_y"] = float(np.mean(np.abs(dev_raw[g0] - y_dev[g0])))
        result["dev_g0_cal_mae_y"] = float(np.mean(np.abs(dev_raw[g0] + b - y_dev[g0])))
        result["dev_cal_mae_y"] = float(np.mean(np.abs(dev_raw + b - y_dev)))
        result["dev_group_counts"] = {
            "k0": int((k_dev == 0).sum()),
            "k-1": int((k_dev == -1).sum()),
            "kle-2": int((k_dev <= -2).sum()),
        }
    np.savez_compressed(out_dir / f"{arm}_predictions.npz", **payload)
    write_json(out_dir / f"{arm}.json", result)
    log(
        f"[{arm}] DONE fit_raw={result['fit_raw_mae_y']:.6f} "
        f"dev_cal={result.get('dev_cal_mae_y', float('nan')):.6f} b={b:.6f} wall={result['wall_clock_s']:.0f}s"
    )
    return result


def _parent_state_hash(model: InterfaceDeploy) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key == "D" or key.startswith("adapter."):
            continue
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _predict(model: nn.Module, data_list: Sequence[Any], targets: torch.Tensor, device: torch.device):
    model.eval()
    preds, tgts = [], []
    state = torch.get_rng_state()
    with torch.no_grad():
        for indices in zftd.epoch_batches(len(data_list), BATCH_SIZE, torch.Generator().manual_seed(0), False):
            batch = zftd.make_batch(data_list, indices, targets, device)
            preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
            tgts.append(batch.y.view(-1).detach().cpu())
    torch.set_rng_state(state)
    return torch.cat(preds).numpy().astype(np.float64), torch.cat(tgts).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# phase 0 — frozen protocol and interface verification
# ---------------------------------------------------------------------------


def phase0(*, device: torch.device = torch.device("cpu"), log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    blob, decomp, train_data, fit_data, dev_data, fit_idx, dev_idx, y = load_data()
    blob_expected_fit = "165e87ef4398ba8ef57411c2f118c4cca4ea74007f2485611b84f0c09bbdd9ea"
    blob_expected_dev = "fb8b78063e7c8a5c759bfa7d553738068cee1738a2d67b210e5d9dc492331376"
    fit_hash = hashlib.sha256(fit_idx.tobytes()).hexdigest()
    dev_hash = hashlib.sha256(dev_idx.tobytes()).hexdigest()
    k = np.asarray(decomp["k"], np.int64)
    counts = {
        "k0": int((k[dev_idx] == 0).sum()),
        "k-1": int((k[dev_idx] == -1).sum()),
        "kle-2": int((k[dev_idx] <= -2).sum()),
    }
    split_ok = bool(
        fit_hash == blob_expected_fit
        and dev_hash == blob_expected_dev
        and counts == {"k0": 1926, "k-1": 65, "kle-2": 9}
    )
    if not split_ok:
        raise RuntimeError(f"frozen split mismatch: {fit_hash} {dev_hash} {counts}")

    structural = load_dictionary_objects()
    stats = compute_interface_stats(
        build_interface_model(structural, {"tau_A": 1.0, "tau_E": 1.0}, code_mode="sparse", block_mode="joint", device=device),
        fit_data,
        device=device,
        log=log,
    )
    stats.update(
        {
            "S_M_raw_soup_sha256": structural["soup_state_sha256"],
            "S_M_source_commit": SOURCE_TRAIN_COMMIT,
            "derivation": "D_star/U/common_rms from the trained S_M soup; kappa re-estimated on 8000 fit; tau from Sparse+Joint blocks (float64)",
        }
    )
    write_json(RESULTS_DIR / "interface_stats.json", stats)

    checks: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "split": {
            "n_fit": int(len(fit_idx)),
            "n_dev": int(len(dev_idx)),
            "fit_idx_sha256": fit_hash,
            "dev_idx_sha256": dev_hash,
            "dev_group_counts": counts,
            "pass": split_ok,
        },
        "sources": {
            "source_branch": SOURCE_BRANCH,
            "source_branch_head": SOURCE_BRANCH_HEAD,
            "source_train_commit": SOURCE_TRAIN_COMMIT,
            "zero_binding_exec_commit": ZERO_BINDING_EXEC_COMMIT,
            "zero_binding_result_commit": ZERO_BINDING_RESULT_COMMIT,
            "S_M_raw_soup_sha256": structural["soup_state_sha256"],
            "S_M_raw_soup_sha256_expected": S_M_SOUP_SHA256,
            "deploy_state_sha256": file_sha256(DEPLOY_STATE),
            "prep_blob_sha256": file_sha256(zjd.PREP_DIR / "fold_objects.npz"),
        },
    }

    # 1. compressed body reproduction (B): deploy state vs full S_M replay
    model = build_interface_model(
        structural, stats, code_mode="sparse", block_mode="joint", device=device
    ).to(device)
    audit = interface_parameter_audit(model)
    replay = np.load(PHASE_A_REPLAY)
    fixed = zftd.make_batch(fit_data, list(range(64)), torch.as_tensor(y[fit_idx][:64], dtype=torch.float32), device)
    ref_dev = replay["dev_raw"][:64].astype(np.float64)
    ref_fit = replay["fit_raw"][:64].astype(np.float64)
    checks["compressed_identity"] = {
        "parameter_audit": audit,
        "adapter_output_absmax_at_init": _adapter_output_absmax(model, fixed),
        "fixed_fit_pred_vs_replay_max_abs": None,
        "dev_replay_file": str(PHASE_A_REPLAY.relative_to(TRACK_ROOT)),
        "note": "dev comparison runs over all 2000 rows below",
        "pass": bool(audit["total_parameters"] == DEPLOY_PARAMETERS + D_PARAMETERS + ADAPTER_PARAMETERS),
    }
    model.eval()
    with torch.no_grad():
        pred_b = model(fixed, mask=cm.C6_MASK).view(-1).detach().cpu().numpy().astype(np.float64)
    checks["compressed_identity"]["fixed_fit_pred_vs_replay_max_abs"] = float(
        np.max(np.abs(pred_b - ref_fit))
    )
    # full dev replay through deploy + interface at init (adapter zero)
    target_dev = torch.as_tensor(y[dev_idx], dtype=torch.float32)
    model_dev = build_interface_model(
        structural, stats, code_mode="sparse", block_mode="joint", device=device
    ).to(device)
    dev_pred, _ = _predict(model_dev, dev_data, target_dev, device)
    checks["compressed_identity"]["full_dev_vs_replay_max_abs"] = float(
        np.max(np.abs(dev_pred - replay["dev_raw"].astype(np.float64)))
    )
    checks["compressed_identity"]["pass"] = bool(
        checks["compressed_identity"]["pass"]
        and checks["compressed_identity"]["fixed_fit_pred_vs_replay_max_abs"] <= COMPRESS_TOL
        and checks["compressed_identity"]["full_dev_vs_replay_max_abs"] <= COMPRESS_TOL
        and checks["compressed_identity"]["adapter_output_absmax_at_init"] <= COMPRESS_TOL
    )

    # 2. operator checks on synthetic micro buckets
    checks["micro_operator"] = _micro_operator_checks(device)

    # 3. real-batch operator checks: J != M, n=1 identity, n=0 zero, order/label
    checks["operator_real"] = _operator_real_checks(model, fit_data, y, fit_idx, device)

    # 4. graph-order / label independence
    checks["invariance"] = _invariance_checks(model, fit_data, y, fit_idx, device)

    # 5. optimizer registration + parent freeze in the smoke path
    checks["registration"] = _registration_checks(structural, stats, fit_data, y, fit_idx, device)

    # 6. schedule hash and arm registry
    schedule = schedule_hash(len(fit_data), PHASE1_EPOCHS, TRAIN_SHUFFLE_OFFSET)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "source_branch": SOURCE_BRANCH,
        "source_branch_head": SOURCE_BRANCH_HEAD,
        "source_train_commit": SOURCE_TRAIN_COMMIT,
        "zero_binding_commits": [ZERO_BINDING_EXEC_COMMIT, ZERO_BINDING_RESULT_COMMIT],
        "arms_phase1": list(PHASE1_ARMS),
        "arm_spec": {k: {"code_mode": v[0], "block_mode": v[1]} for k, v in ARM_SPEC.items()},
        "arms_phase2": list(PHASE2_ARMS),
        "seed": SEED,
        "scale_seed": SCALE_SEED,
        "adapter_seed": ADAPTER_SEED,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": BASE_EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "h1_lambda": H1_LAMBDA,
        "soup_epochs_phase1": [156, 157, 158, 159, 160],
        "soup_epochs_phase2": [236, 237, 238, 239, 240],
        "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
        "schedule_phase1": schedule,
        "adapter_in": ADAPTER_IN,
        "adapter_hidden": ADAPTER_HIDDEN,
        "fusion_out": FUSION_OUT,
        "expected_parameters": {
            "body": DEPLOY_PARAMETERS,
            "D": D_PARAMETERS,
            "adapter": ADAPTER_PARAMETERS,
            "total": DEPLOY_PARAMETERS + D_PARAMETERS + ADAPTER_PARAMETERS,
        },
        "interface_stats": stats,
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "delta": DELTA},
        "primary_endpoint": "dev G0 (k=0) cal MAE",
        "secondary_endpoint": "dev overall cal MAE",
        "split": {
            "n_fit": int(len(fit_idx)),
            "n_dev": int(len(dev_idx)),
            "fit_idx_sha256": fit_hash,
            "dev_idx_sha256": dev_hash,
            "dev_group_counts": counts,
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(RESULTS_DIR / "protocol.json", protocol)
    checks["seconds"] = float(time.perf_counter() - t0)
    checks["pass"] = bool(
        checks["compressed_identity"]["pass"]
        and checks["micro_operator"]["pass"]
        and checks["operator_real"]["pass"]
        and checks["invariance"]["pass"]
        and checks["registration"]["pass"]
    )
    write_json(RESULTS_DIR / "phase0_checks.json", checks)
    log(f"[phase0] pass={checks['pass']} seconds={checks['seconds']:.1f}")
    return checks


def _adapter_output_absmax(model: InterfaceDeploy, batch: Any) -> float:
    was = model.training
    model.eval()
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        interface = sem.SEM108Model.semantic_interface(model, coord, batch, cm.C6_MASK, None)
        delta = model.adapter(model.adapter_input(interface, coord, batch))
    model.train(was)
    return float(delta.abs().max())


def _micro_operator_checks(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(7)
    checks: dict[str, Any] = {}
    # node: 2 occurrences, same bucket; permutation of atom categories
    a = torch.randn(2, K_ATOMS, generator=gen)
    t = torch.zeros(2, ATOM_CATEGORIES)
    t[0, 0] = 1.0
    t[1, 1] = 1.0

    class _D:
        pass

    # build a minimal data stub for joint_node_block
    data = _D()
    data.env_occ_root = torch.tensor([0, 0])
    data.env_occ_shell = torch.tensor([0, 0])
    data.env_occ_node = torch.tensor([0, 1])
    data.dict_atom = torch.tensor([0, 1])
    j1 = joint_node_block(a, data, "joint")
    m1 = joint_node_block(a, data, "marginal")
    data_perm = _D()
    data_perm.env_occ_root = torch.tensor([0, 0])
    data_perm.env_occ_shell = torch.tensor([0, 0])
    data_perm.env_occ_node = torch.tensor([0, 1])
    data_perm.dict_atom = torch.tensor([1, 0])
    j2 = joint_node_block(a, data_perm, "joint")
    m2 = joint_node_block(a, data_perm, "marginal")
    # n=1 identity
    data1 = _D()
    data1.env_occ_root = torch.tensor([0])
    data1.env_occ_shell = torch.tensor([0])
    data1.env_occ_node = torch.tensor([0])
    data1.dict_atom = torch.tensor([0])
    jn1 = joint_node_block(a[:1], data1, "joint")
    mn1 = joint_node_block(a[:1], data1, "marginal")
    # n=0 zero
    data0 = _D()
    data0.env_occ_root = torch.zeros(0, dtype=torch.long)
    data0.env_occ_shell = torch.zeros(0, dtype=torch.long)
    data0.env_occ_node = torch.zeros(0, dtype=torch.long)
    data0.dict_atom = torch.zeros(0, dtype=torch.long)
    node0 = joint_node_block(a[:0], data0, "marginal")
    checks["node"] = {
        "J_changes_under_atom_permutation": float((j1 - j2).abs().max()),
        "M_invariant_under_atom_permutation": float((m1 - m2).abs().max()),
        "n1_J_equals_M": float((jn1 - mn1).abs().max()),
        "n0_M_zero": float(node0.abs().max()) if node0.numel() else 0.0,
    }
    # edge analogue: two occurrences with distinct endpoint pairs so J is pairing-sensitive
    a3 = torch.randn(3, K_ATOMS, generator=gen)
    de = _D()
    de.env_bond_root = torch.tensor([0, 0])
    de.env_bond_shellpair = torch.tensor([0, 0])
    de.env_bond_u = torch.tensor([0, 0])
    de.env_bond_v = torch.tensor([1, 2])
    de.env_bond_type = torch.tensor([0, 1])
    je1 = joint_edge_block(a3, de, "joint")
    me1 = joint_edge_block(a3, de, "marginal")
    de2 = _D()
    de2.env_bond_root = torch.tensor([0, 0])
    de2.env_bond_shellpair = torch.tensor([0, 0])
    de2.env_bond_u = torch.tensor([0, 0])
    de2.env_bond_v = torch.tensor([1, 2])
    de2.env_bond_type = torch.tensor([1, 0])
    je2 = joint_edge_block(a3, de2, "joint")
    me2 = joint_edge_block(a3, de2, "marginal")
    checks["edge"] = {
        "J_changes_under_type_permutation": float((je1 - je2).abs().max()),
        "M_invariant_under_type_permutation": float((me1 - me2).abs().max()),
    }
    checks["pass"] = bool(
        checks["node"]["J_changes_under_atom_permutation"] > 0
        and checks["node"]["M_invariant_under_atom_permutation"] <= 1e-6
        and checks["node"]["n1_J_equals_M"] <= 1e-6
        and checks["node"]["n0_M_zero"] == 0.0
        and checks["edge"]["J_changes_under_type_permutation"] > 0
        and checks["edge"]["M_invariant_under_type_permutation"] <= 1e-6
    )
    return checks


def _operator_real_checks(
    model: InterfaceDeploy,
    fit_data: Sequence[Any],
    y: np.ndarray,
    fit_idx: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    batch = zftd.make_batch(
        fit_data, list(range(8)), torch.as_tensor(y[fit_idx][:8], dtype=torch.float32), device
    )
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        alpha = coord[:, model.common_dim :]
        node_j = joint_node_block(alpha, batch, "joint")
        node_m = joint_node_block(alpha, batch, "marginal")
        edge_j = joint_edge_block(alpha, batch, "joint")
        edge_m = joint_edge_block(alpha, batch, "marginal")
        delta_node = (node_j - node_m).abs()
        delta_edge = (edge_j - edge_m).abs()
        cnt_n = torch.zeros(int(node_j.shape[0]))
        cnt_n.index_add_(0, batch.env_occ_root.to(device) * N_SHELLS + batch.env_occ_shell.to(device), torch.ones_like(batch.env_occ_root, dtype=torch.float32, device=device))
        cnt_e = torch.zeros(int(edge_j.shape[0]))
        cnt_e.index_add_(0, batch.env_bond_root.to(device) * SHELLPAIR_CLASSES + batch.env_bond_shellpair.to(device), torch.ones_like(batch.env_bond_root, dtype=torch.float32, device=device))
    return {
        "node_J_neq_M_max_abs": float(delta_node.max()),
        "edge_J_neq_M_max_abs": float(delta_edge.max()),
        "node_bucket_ge2_fraction": float((cnt_n >= 2).float().mean()),
        "edge_bucket_ge2_fraction": float((cnt_e >= 2).float().mean()),
        "node_finite": bool(torch.isfinite(node_j).all() and torch.isfinite(node_m).all()),
        "edge_finite": bool(torch.isfinite(edge_j).all() and torch.isfinite(edge_m).all()),
        "pass": bool(
            float(delta_node.max()) > 0
            and float(delta_edge.max()) > 0
            and bool(torch.isfinite(node_j).all())
            and bool(torch.isfinite(edge_j).all())
        ),
    }


def _invariance_checks(model: InterfaceDeploy, fit_data, y, fit_idx, device) -> dict[str, Any]:
    model.eval()
    y_fit = torch.as_tensor(y[fit_idx], dtype=torch.float32)
    batch = zftd.make_batch(fit_data, [0, 1], y_fit, device)
    with torch.no_grad():
        p_ab = model(batch, mask=cm.C6_MASK).view(-1).clone()
        batch_ba = zftd.make_batch(fit_data, [1, 0], y_fit, device)
        p_ba = model(batch_ba, mask=cm.C6_MASK).view(-1).clone()
        batch2 = batch.clone()
        batch2.y = torch.randn_like(batch2.y)
        p_shuf = model(batch2, mask=cm.C6_MASK).view(-1).clone()
    order = float(max(abs(float(p_ab[0]) - float(p_ba[1])), abs(float(p_ab[1]) - float(p_ba[0]))))
    return {
        "graph_order_map_diff": order,
        "label_shuffle_max_abs": float((p_ab - p_shuf).abs().max()),
        "pass": bool(order <= 1.0e-4 and float((p_ab - p_shuf).abs().max()) <= COMPRESS_TOL),
    }


def _registration_checks(structural, stats, fit_data, y, fit_idx, device) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    y_fit = torch.as_tensor(y[fit_idx][:16], dtype=torch.float32)
    # Phase 1: only the adapter is a trainable parameter; two coupled steps show
    # the standard zero-init behaviour (step-1 upstream grad zero, step-2 alive).
    model = build_interface_model(structural, stats, code_mode="sparse", block_mode="joint", device=device).to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("adapter."))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    batch = zftd.make_batch(fit_data, list(range(16)), y_fit, device)
    names = dict(model.named_parameters())
    step_grads = {}
    for step in (1, 2):
        model.train()
        prediction, _aux = model(batch, mask=cm.C6_MASK, return_aux=True)
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        step_grads[f"step{step}"] = {
            "adapter0_weight_absmax": float(names["adapter.0.weight"].grad.abs().max()),
            "adapter2_weight_absmax": float(names["adapter.2.weight"].grad.abs().max()),
        }
        optimizer.step()
    checks["R_SJ_phase1"] = {
        "optimizer_parameters": int(sum(p.numel() for p in trainable)),
        "expected_adapter_parameters": ADAPTER_PARAMETERS,
        "adapter_only_trainable": bool(
            all(p.requires_grad == n.startswith("adapter.") for n, p in model.named_parameters())
        ),
        "step_grads": step_grads,
        "step1_upstream_zero": bool(step_grads["step1"]["adapter0_weight_absmax"] == 0.0),
        "step2_upstream_alive": bool(step_grads["step2"]["adapter0_weight_absmax"] > 0.0),
    }
    checks["R_SJ_phase1"]["pass"] = bool(
        checks["R_SJ_phase1"]["optimizer_parameters"] == ADAPTER_PARAMETERS
        and checks["R_SJ_phase1"]["adapter_only_trainable"]
        and checks["R_SJ_phase1"]["step1_upstream_zero"]
        and checks["R_SJ_phase1"]["step2_upstream_alive"]
    )

    # Phase 2: all parameters registered; D gets the reconstruction gradient and
    # the adapter gets task gradients.
    model2 = build_interface_model(structural, stats, code_mode="sparse", block_mode="joint", device=device).to(device)
    for parameter in model2.parameters():
        parameter.requires_grad_(True)
    trainable2 = [p for p in model2.parameters() if p.requires_grad]
    batch = zftd.make_batch(fit_data, list(range(16)), y_fit, device)
    model2.train()
    prediction, aux = model2(batch, mask=cm.C6_MASK, return_aux=True)
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + H1_LAMBDA * model2.reconstruction_loss(aux["phi"], aux["coord"])
    model2.zero_grad(set_to_none=True)
    loss.backward()
    names2 = dict(model2.named_parameters())
    checks["A0_phase2"] = {
        "optimizer_parameters": int(sum(p.numel() for p in trainable2)),
        "expected_total": DEPLOY_PARAMETERS + D_PARAMETERS + ADAPTER_PARAMETERS,
        "D_grad_absmax": float(names2["D"].grad.abs().max()),
        "adapter2_weight_grad_absmax": float(names2["adapter.2.weight"].grad.abs().max()),
        "reconstruction_grad_alive": bool(float(names2["D"].grad.abs().max()) > 0),
    }
    checks["A0_phase2"]["pass"] = bool(
        checks["A0_phase2"]["optimizer_parameters"] == DEPLOY_PARAMETERS + D_PARAMETERS + ADAPTER_PARAMETERS
        and checks["A0_phase2"]["reconstruction_grad_alive"]
    )
    model2.zero_grad(set_to_none=True)
    checks["pass"] = bool(checks["R_SJ_phase1"]["pass"] and checks["A0_phase2"]["pass"])
    return checks


# ---------------------------------------------------------------------------
# smoke (engineering only)
# ---------------------------------------------------------------------------


def run_smoke(*, device: torch.device, out_dir: Path = RESULTS_DIR, max_steps: int = 4, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    blob, decomp, train_data, fit_data, dev_data, fit_idx, dev_idx, y = load_data()
    structural = load_dictionary_objects()
    stats = load_interface_stats()
    checks: dict[str, Any] = {"official_valid_loaded": False, "official_test_loaded": False}
    y_fit = np.asarray(y[fit_idx], np.float64)

    # shared init across all five phase-1 arms
    inits = {}
    for arm, (code_mode, block_mode) in ARM_SPEC.items():
        seed_everything(SEED)
        model = build_interface_model(structural, stats, code_mode=code_mode, block_mode=block_mode, device=device).to(device)
        inits[arm] = {
            "adapter_init_hash": _adapter_hash(model),
            "body_hash": _parent_state_hash(model),
            "D_hash": zsf.state_hash({"D": model.D.detach().cpu()}),
        }
    checks["shared_init"] = {
        "adapter_hashes": {k: v["adapter_init_hash"] for k, v in inits.items()},
        "body_hashes": {k: v["body_hash"] for k, v in inits.items()},
        "D_hashes": {k: v["D_hash"] for k, v in inits.items()},
        "pass": bool(
            len({v["adapter_init_hash"] for v in inits.values()}) == 1
            and len({v["body_hash"] for v in inits.values()}) == 1
            and len({v["D_hash"] for v in inits.values()}) == 1
        ),
    }

    # phase 1: R_SJ and R_CS smoke, state discarded
    smoke1 = {}
    for arm in ("R_CS", "R_SJ"):
        code_mode, block_mode = ARM_SPEC[arm]
        res = train_arm(
            arm,
            phase=1,
            code_mode=code_mode,
            block_mode=block_mode,
            device=device,
            out_dir=out_dir / "smoke",
            epochs=1,
            max_steps=int(max_steps),
            log=None,
        )
        smoke1[arm] = {
            "steps_done": res["steps_done"],
            "init_hash": res["init_trainable_hash"],
            "last_hash": res["last_trainable_hash"],
            "parent_unchanged": bool(res["parent_init_state_hash"] == res["parent_last_state_hash"]),
            "curve": res["curve"],
        }
    # phase 2: A0 / C / T smoke
    smoke2 = {}
    for arm in PHASE2_ARMS:
        code_mode, block_mode = ("sparse", "joint") if arm != "C" else ("sparse", "joint")
        if arm == "C":
            block_mode = "none"
        res = train_arm(
            arm,
            phase=2,
            code_mode=code_mode,
            block_mode=block_mode,
            device=device,
            out_dir=out_dir / "smoke",
            epochs=1,
            max_steps=int(max_steps),
            log=None,
        )
        smoke2[arm] = {
            "steps_done": res["steps_done"],
            "train_rec": res["curve"][0]["train_rec"],
            "parent_unchanged": bool(res["parent_init_state_hash"] == res["parent_last_state_hash"]),
        }
    checks["phase1_smoke"] = smoke1
    checks["phase2_smoke"] = smoke2
    checks["pass"] = bool(
        checks["shared_init"]["pass"]
        and all(v["steps_done"] == int(max_steps) and v["parent_unchanged"] for v in smoke1.values())
        and all(v["steps_done"] == int(max_steps) and v["train_rec"] > 0 for v in smoke2.values())
    )
    write_json(out_dir / "smoke" / "smoke.json", checks)
    log(f"[smoke] pass={checks['pass']}")
    return checks


def _adapter_hash(model: InterfaceDeploy) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.named_parameters()):
        if not key.startswith("adapter."):
            continue
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZINC overnight interface + tail seed0 v1 (main)")
    parser.add_argument("--mode", required=True, choices=("phase0", "smoke", "phase1", "phase2", "phase3"))
    parser.add_argument("--arm", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "phase0":
        checks = phase0(device=resolve_device(args.device))
        return 0 if checks["pass"] else 1
    if args.mode == "smoke":
        checks = run_smoke(device=resolve_device(args.device), out_dir=out_dir, max_steps=int(args.max_steps or 4))
        return 0 if checks["pass"] else 1

    device = resolve_device(args.device)
    if args.mode == "phase1":
        assert args.arm in ARM_SPEC, f"phase1 arm must be one of {list(ARM_SPEC)}"
        code_mode, block_mode = ARM_SPEC[args.arm]
        epochs = int(args.epochs or PHASE1_EPOCHS)
        train_arm(
            args.arm,
            phase=1,
            code_mode=code_mode,
            block_mode=block_mode,
            device=device,
            out_dir=out_dir,
            epochs=epochs,
            max_steps=args.max_steps,
        )
        return 0
    if args.mode in ("phase2", "phase3"):
        assert args.arm in PHASE2_ARMS, f"phase2/3 arm must be one of {PHASE2_ARMS}"
        # The mode (sparse/dense) and the interface block come from the Phase 1
        # winner T, frozen by the decision file before Phase 2 is launched.
        decision = json.loads((RESULTS_DIR / "stage2_decision.json").read_text())
        spec = decision["selected_spec"]
        code_mode = str(spec["code_mode"])
        if args.arm == "C":
            block_mode = "none"
        else:
            block_mode = str(spec["block_mode"])
        epochs = int(args.epochs or BASE_EPOCHS)
        train_arm(
            args.arm,
            phase=2 if args.mode == "phase2" else 2,
            code_mode=code_mode,
            block_mode=block_mode,
            device=device,
            out_dir=out_dir,
            epochs=epochs,
            max_steps=args.max_steps,
            train_all=bool(args.mode == "phase3"),
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
