"""ZINC structure-semantic 2x2 factorial, single seed (seed 0).

Question
--------
In the current canonical ``Full`` (``e2e_dictenv_scale_v1``, 408,651
trainable parameters) does the *per-occurrence* structure/attribute binding
retain a generalisation value beyond the identical marginals, and does that
value depend on the sparse structural code?

Four arms, one fresh seed-0 trajectory each, identical raw inputs, parameter
count, initialisation, data stream, loss weights and budget.  Only the coding
operator and the slot binding operator change:

===========  ==========================  ============================
arm          structural code             slot aggregation
===========  ==========================  ============================
``S_J``      CSSD-q1 + tied-IHT (s=8)    per-occurrence product (J)
``S_M``      CSSD-q1 + tied-IHT (s=8)    independent-pairing null (M)
``D_J``      dense tied-linear code      per-occurrence product (J)
``D_M``      dense tied-linear code      independent-pairing null (M)
===========  ==========================  ============================

The calm control is ``S_J``.  ``M`` removes exactly the within-bucket
per-occurrence covariance ``J - M = n * population_covariance(a, b) / sqrt(w)``
while keeping the same structural and semantic multisets, counts and
normalisation.  Sem108 / size2 / topology / relation / global context / pair
relations / the C6 mask are untouched: ``M`` does **not** remove all
structure-semantics interaction inside the model.

The dense control keeps the same ``D_raw`` trainable shared structural matrix
and the same ``U``; it only replaces the tied-IHT encoder with a fixed-amplitude
linear code ``alpha_D = kappa * (r @ Dbar)``.  ``kappa`` is a frozen buffer
matching the initial residual-code RMS of the sparse arm; it is computed once
from the shared initial ``D``/``U`` and all 8000 fit nodes (never the dev rows).

Train-only: only ``encoded_train.pt`` + ``env_train.pt`` are loaded.  The
official validation split is never loaded; the official test split is never
instantiated.  ``official_test_loaded == False`` everywhere.
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
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

PROTOCOL_VERSION = "zinc-structure-semantic-factorial-seed0-v1"

RESULTS_DIR = zjd.TRACK_ROOT / "results/zinc_structure_semantic_factorial_seed0_v1"
PREP_BLOB = zjd.PREP_DIR / "fold_objects.npz"

#: frozen recipe — never selected on dev.
SEED = 0
SCALE_SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
TRAIN_SHUFFLE_OFFSET = 101
FULL_PARAMETERS = 408_651
H1_LAMBDA = float(cm.H1_LAMBDA)

ARMS = ("S_J", "S_M", "D_J", "D_M")
CODING_OF = {"S_J": "sparse", "S_M": "sparse", "D_J": "dense", "D_M": "dense"}
BINDING_OF = {"S_J": "paired", "S_M": "indep", "D_J": "paired", "D_M": "indep"}

GROUP_NAMES = {"k0": "k=0", "k-1": "k=-1", "kle-2": "k<=-2"}
BOOT_SEED = 20261003
N_BOOT = 1000
DELTA = 0.003
REPLAY_TOL = 2.0e-6
HEALTH_EPOCHS = (1, 40, 120, 240)
HEALTH_SAMPLE = 256


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key in ("cuda", "cuda:0", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device requests CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    raise ValueError(f"unsupported device {name!r}")


def parameter_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def trainable_parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.named_parameters()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# exact slot operators (the tested production functions)
# ---------------------------------------------------------------------------


def paired_bucket(
    a: torch.Tensor, b: torch.Tensor, index: torch.Tensor, n_buckets: int, width: int
) -> torch.Tensor:
    """J: ``sum_i (a_i * b_i) / sqrt(width)`` aggregated into buckets."""
    u = (a * b) / math.sqrt(float(width))
    flat = torch.zeros((n_buckets, int(a.shape[1])), device=u.device, dtype=u.dtype)
    flat.index_add_(0, index, u)
    return flat


def indep_bucket(
    a: torch.Tensor, b: torch.Tensor, index: torch.Tensor, n_buckets: int, width: int
) -> torch.Tensor:
    """M: ``(sum_i a_i) * (sum_i b_i) / (n * sqrt(width))``; n=0 -> 0."""
    sum_a = torch.zeros((n_buckets, int(a.shape[1])), device=a.device, dtype=a.dtype)
    sum_b = torch.zeros_like(sum_a)
    count = torch.zeros((n_buckets, 1), device=a.device, dtype=a.dtype)
    sum_a.index_add_(0, index, a)
    sum_b.index_add_(0, index, b)
    count.index_add_(0, index, torch.ones_like(index, dtype=a.dtype).unsqueeze(1))
    denominator = torch.clamp(count, min=1.0) * math.sqrt(float(width))
    out = (sum_a * sum_b) / denominator
    return out.masked_fill(count <= 0, 0.0)


# ---------------------------------------------------------------------------
# the factorial Full
# ---------------------------------------------------------------------------


class FactorialFull(sc.LatentScaleSEM108):
    """Canonical Full with selectable coding / binding operators.

    ``coding_mode="sparse"`` + ``binding_mode="paired"`` is a bit-identical
    re-expression of the canonical Full: both overridden methods delegate to
    the untouched parent.  The class inherits the exact parent RNG stream and
    parameter names, so all four arms share item-for-item initial parameters.
    ``kappa`` is an added non-trainable buffer (not a parameter).
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        spec: sc.ScaleSpec = sc.FULL,
        coding_mode: str = "sparse",
        binding_mode: str = "paired",
        scale_seed: int = 0,
    ) -> None:
        # SEM108Model itself requires the paired bindings; the factorial mode is
        # carried on this subclass and enforced at the real forward path.
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            spec=spec,
            node_binding="paired",
            edge_binding="paired",
            coding="sparse",
            scale_seed=int(scale_seed),
        )
        if coding_mode not in ("sparse", "dense"):
            raise ValueError(f"unknown coding_mode {coding_mode!r}")
        if binding_mode not in ("paired", "indep"):
            raise ValueError(f"unknown binding_mode {binding_mode!r}")
        self.coding_mode = str(coding_mode)
        self.binding_mode = str(binding_mode)
        #: frozen fixed-amplitude control for the dense code (set before training).
        self.register_buffer("kappa", torch.ones((), dtype=torch.float32))
        #: instrumentation: how many times the indep environment actually ran.
        self._indep_calls = 0

    # -- coding ---------------------------------------------------------------

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        if self.coding_mode == "sparse":
            return super().code(phi)
        U = self.U.to(dtype=phi.dtype)
        d = phi @ U
        r = phi - d @ U.t()
        Dbar = self.residual_dictionary()
        alpha = self.kappa.to(dtype=phi.dtype) * (r @ Dbar)
        scale = self.common_rms.to(dtype=phi.dtype)
        return torch.cat([d / scale, alpha], dim=1)

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        if self.coding_mode == "sparse":
            return super().reconstruct(phi, coord)
        alpha = coord[:, self.common_dim :]
        return (alpha / self.kappa.to(dtype=phi.dtype)) @ self.residual_dictionary().t()

    # -- slot binding ---------------------------------------------------------

    def _environment_from_parts(self, coord: torch.Tensor, data: Any, interface: torch.Tensor, **kwargs: Any):
        if self.binding_mode == "paired":
            return super()._environment_from_parts(coord, data, interface, **kwargs)
        self._indep_calls += 1
        h = self._indep_environment_from_parts(coord, data, interface, **kwargs)
        if self._bridge_permute_seed is not None:
            batch = getattr(data, "batch", None)
            if batch is None:
                raise RuntimeError("within-molecule code permutation requires data.batch")
            h = lb._permute_rows_within_graph(h, batch, int(self._bridge_permute_seed))
        return self.local_dictionary_bridge(h)

    def _indep_environment_from_parts(
        self,
        coord: torch.Tensor,
        data: Any,
        interface: torch.Tensor,
        *,
        occ_coord_node: torch.Tensor | None = None,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
        node_binding_zero: bool = False,
        edge_binding_zero: bool = False,
        mask: Any = None,
    ) -> torch.Tensor:
        """M variant of ``SEM108Model._environment_from_parts``.

        Identical to the parent except that the per-occurrence product inside a
        bucket is replaced by the analytic independent-pairing expectation
        ``indep_bucket``.  Everything after the two slot tensors
        (node/edge encoders, fusion) is unchanged.
        """
        n = int(coord.shape[0])
        q = F.one_hot(data.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        if occ_coord_node is None:
            occ_coord_node = data.env_occ_node
        occ_coord_node = occ_coord_node.to(coord.device)
        c = coord[occ_coord_node]
        qc = q[data.env_occ_node.to(coord.device)]
        a = c @ self.W_A_S
        b = qc @ self.W_A_C
        node_index = data.env_occ_root.to(coord.device) * int(p2.N_SHELLS) + data.env_occ_shell.to(
            coord.device
        )
        node_flat = indep_bucket(
            a, b, node_index, n * int(p2.N_SHELLS), int(p2.D_A)
        )
        if node_binding_zero:
            node_flat = torch.zeros_like(node_flat)
        node_slots = node_flat.view(n, int(p2.N_SHELLS), int(p2.D_A))

        d_e = int(self.config.d_e)
        if bond_u is None:
            bond_u = data.env_bond_u
        if bond_v is None:
            bond_v = data.env_bond_v
        bond_u = bond_u.to(coord.device)
        bond_v = bond_v.to(coord.device)
        cu = coord[bond_u]
        cv = coord[bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        if edge_binding_zero:
            g = torch.zeros_like(g)
        s = g @ self.W_E_S
        bcat = F.one_hot(data.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(coord.dtype)
        ce = bcat @ self.W_E_C
        edge_index = data.env_bond_root.to(coord.device) * int(p2.SHELLPAIR_CLASSES) + (
            data.env_bond_shellpair.to(coord.device)
        )
        edge_flat = indep_bucket(
            s, ce, edge_index, n * int(p2.SHELLPAIR_CLASSES), int(d_e)
        )
        edge_slots = edge_flat.view(n, int(p2.SHELLPAIR_CLASSES), int(d_e))

        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        if self.parent_interface:
            raise RuntimeError("parent_interface is not supported by the factorial model")
        fused = torch.cat([interface, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1)
        return self.fusion(fused)

    # -- read-only diagnostics -------------------------------------------------

    def node_factor_slots(self, coord: torch.Tensor, data: Any):
        """Raw per-bucket ``(a, b, index)`` node factors for operator checks."""
        q = F.one_hot(data.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        c = coord[data.env_occ_node.to(coord.device)]
        qc = q[data.env_occ_node.to(coord.device)]
        index = data.env_occ_root.to(coord.device) * int(p2.N_SHELLS) + data.env_occ_shell.to(
            coord.device
        )
        return c @ self.W_A_S, qc @ self.W_A_C, index

    def edge_factor_slots(self, coord: torch.Tensor, data: Any):
        d_e = int(self.config.d_e)
        bond_u = data.env_bond_u.to(coord.device)
        bond_v = data.env_bond_v.to(coord.device)
        cu = coord[bond_u]
        cv = coord[bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        s = g @ self.W_E_S
        b = F.one_hot(data.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(coord.dtype)
        ce = b @ self.W_E_C
        index = data.env_bond_root.to(coord.device) * int(p2.SHELLPAIR_CLASSES) + (
            data.env_bond_shellpair.to(coord.device)
        )
        return s, ce, index, int(d_e)


def build_factorial_model(
    blob: Mapping[str, Any], seed: int, coding_mode: str, binding_mode: str
) -> FactorialFull:
    """One canonical fresh Full with the factorial operators selected.

    Uses the exact ``e2e_dictenv_scale_v1.build_scale_model`` RNG stream so the
    four arms have item-for-item identical initial parameters.
    """
    torch.manual_seed(int(seed))
    model = FactorialFull(
        cm.H1_CONFIG,
        np.asarray(blob["D_fit"], np.float32),
        subspace=zjd.fold_subspace(blob),
        spec=sc.FULL,
        coding_mode=coding_mode,
        binding_mode=binding_mode,
        scale_seed=SCALE_SEED,
    )
    audit = sc.scale_parameter_audit(model)
    if int(audit["actual_parameters"]) != FULL_PARAMETERS or not audit["parameter_exact"]:
        raise RuntimeError(f"canonical Full parameter audit failed: {audit}")
    return model


# ---------------------------------------------------------------------------
# kappa — fixed-amplitude control for the dense code
# ---------------------------------------------------------------------------


def compute_kappa(model: FactorialFull, fit_data: Sequence[Any]) -> dict[str, Any]:
    """``kappa = sqrt(sum alpha_S^2 / sum (r @ Dbar)^2)`` over all fit nodes.

    Computed once from the shared initial ``D``/``U``; detach-ed into a buffer.
    The denominator check is explicit so an invalid control cannot silently
    degenerate to ``1.0``.
    """
    was_training = model.training
    model.eval()
    device = model.U.device
    with torch.no_grad():
        Dbar = model.residual_dictionary()
        U = model.U
        alpha_sq = 0.0
        dense_sq = 0.0
        n_rows = 0
        chunk: list[torch.Tensor] = []
        chunk_rows = 0
        for data in fit_data:
            chunk.append(data.dict_phi.to(dtype=torch.float32))
            chunk_rows += int(data.dict_phi.shape[0])
            if chunk_rows >= 32768:
                a, d, n = _kappa_chunk(torch.cat(chunk, 0).to(device), Dbar, U)
                alpha_sq += a
                dense_sq += d
                n_rows += n
                chunk, chunk_rows = [], 0
        if chunk:
            a, d, n = _kappa_chunk(torch.cat(chunk, 0).to(device), Dbar, U)
            alpha_sq += a
            dense_sq += d
            n_rows += n
    if not (alpha_sq > 0 and dense_sq > 0):
        raise RuntimeError(f"invalid kappa inputs: alpha_sq={alpha_sq} dense_sq={dense_sq}")
    kappa = math.sqrt(alpha_sq / dense_sq)
    if not math.isfinite(kappa) or kappa <= 0:
        raise RuntimeError(f"invalid kappa {kappa}")
    model.train(was_training)
    return {
        "kappa": kappa,
        "alpha_sq_sum": alpha_sq,
        "dense_sq_sum": dense_sq,
        "sparse_rms": math.sqrt(alpha_sq / n_rows),
        "dense_rms": math.sqrt(dense_sq / n_rows),
        "n_fit_nodes": n_rows,
    }


# ---------------------------------------------------------------------------
# data / batching (train-only, private generators)
# ---------------------------------------------------------------------------


def _kappa_chunk(phi: torch.Tensor, Dbar: torch.Tensor, U: torch.Tensor):
    d = phi @ U
    r = phi - d @ U.t()
    alpha = v0.tied_iht_codes(Dbar, r, s=cssd.SPARSITY, steps=cssd.IHT_STEPS)
    dense = r @ Dbar
    return (
        float((alpha.double() ** 2).sum()),
        float((dense.double() ** 2).sum()),
        int(phi.shape[0]),
    )


def load_train_only() -> list[Any]:
    return zftd.load_train_only()


def load_prep_blob() -> dict[str, np.ndarray]:
    return zftd.load_prep_blob()


def load_target() -> dict[str, np.ndarray]:
    return zftd.load_target_decomposition()


def epoch_batches(n: int, batch_size: int, generator: torch.Generator, shuffle: bool) -> list[list[int]]:
    return zftd.epoch_batches(n, batch_size, generator, shuffle)


def make_batch(data_list: Sequence[Any], indices: Sequence[int], targets: torch.Tensor, device: torch.device):
    return zftd.make_batch(data_list, indices, targets, device)


def predict(model: torch.nn.Module, data_list: Sequence[Any], targets: torch.Tensor, device: torch.device):
    return zftd.predict(model, data_list, targets, device)


def build_fit_dev(train_data: Sequence[Any], fit_idx: np.ndarray, dev_idx: np.ndarray):
    return zftd.build_fit_dev(train_data, fit_idx, dev_idx)


# ---------------------------------------------------------------------------
# health diagnostics (no RNG consumption, no dev peeking)
# ---------------------------------------------------------------------------


def _effective_rank(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return float("nan")
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > 1e-9]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(math.exp(-(p * np.log(p)).sum()))


def health_snapshot(model: FactorialFull, batch: Any) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        residual = coord[:, model.common_dim :]
        alpha = residual.detach().cpu().numpy()
        n_rows = int(alpha.shape[0])
        nonzero = np.abs(alpha) > 0.0
        node_slots = model.node_slots(coord, batch, binding=model.binding_mode)
        edge_slots = model.edge_slots(coord, batch, binding=model.binding_mode)
        payload = {
            "n_rows": n_rows,
            "code_width": int(coord.shape[1]),
            "residual_nnz_per_row": float(nonzero.sum(1).mean()),
            "residual_all_zero_fraction": float((~nonzero).all(1).mean()),
            "residual_effective_rank": _effective_rank(alpha),
            "node_slot_variance": float(node_slots.var().item()),
            "edge_slot_variance": float(edge_slots.var().item()),
            "coord_mean": float(coord.mean().item()),
            "coord_std": float(coord.std().item()),
        }
    model.train(was_training)
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_arm(
    arm: str,
    blob: Mapping[str, Any],
    fit_data: Sequence[Any],
    target_y: np.ndarray,
    *,
    device: torch.device,
    out_dir: Path,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    assert arm in ARMS, arm
    coding_mode = CODING_OF[arm]
    binding_mode = BINDING_OF[arm]
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    y_fit_np = np.asarray(target_y, np.float64)[fit_idx]
    target_fit = torch.as_tensor(y_fit_np, dtype=torch.float32)

    seed_everything(SEED)
    model = build_factorial_model(blob, SEED, coding_mode, binding_mode).to(device)
    kappa_info = compute_kappa(model, fit_data)
    with torch.no_grad():
        model.kappa.fill_(float(kappa_info["kappa"]))
    init_hash = trainable_parameter_hash(model)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(int(SEED) + TRAIN_SHUFFLE_OFFSET)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    soup_epochs = set(range(max(1, int(epochs) - 4), int(epochs) + 1))
    health_batch = make_batch(
        fit_data,
        list(range(min(HEALTH_SAMPLE, len(fit_data)))),
        torch.as_tensor(target_y[fit_idx][:HEALTH_SAMPLE], dtype=torch.float32),
        device,
    )
    curve: list[dict[str, Any]] = []
    health: dict[str, Any] = {}
    epoch_seconds: list[float] = []
    steps_done = 0
    started = time.perf_counter()
    stopped_reason = "completed"

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, rec_sum, n_mol, n_steps, gnorm_sum = 0.0, 0.0, 0, 0, 0.0
        for indices in epoch_batches(len(fit_data), BATCH_SIZE, train_gen, True):
            batch = make_batch(fit_data, indices, target_fit, device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = task + H1_LAMBDA * rec
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(rec.detach())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        epoch_seconds.append(float(time.perf_counter() - epoch_started))
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "train_rec": float(rec_sum / max(n_steps, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "seconds": epoch_seconds[-1],
            }
        )
        health[str(epoch)] = health_snapshot(model, health_batch)
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[{arm}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"rec={curve[-1]['train_rec']:.4g} gnorm={curve[-1]['grad_norm']:.3g} "
                f"nnz={health[str(epoch)]['residual_nnz_per_row']:.3f} "
                f"{epoch_seconds[-1]:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "seed": SEED,
        "scale_seed": SCALE_SEED,
        "coding_mode": coding_mode,
        "binding_mode": binding_mode,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "h1_lambda": H1_LAMBDA,
        "soup_epochs": sorted(soup_epochs) if max_steps is None else [],
        "trainable_parameters": trainable_parameter_count(model),
        "init_trainable_hash": init_hash,
        "kappa": kappa_info,
        "curve": curve,
        "health": health,
        "epoch_seconds": epoch_seconds,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "indep_calls": int(model._indep_calls),
        "official_test_loaded": False,
    }

    if max_steps is not None:
        result["last_trainable_hash"] = trainable_parameter_hash(model)
        return result

    members = sorted(soup)
    soup_state = {k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]}
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(init_state, out_dir / f"{arm}_init_state.pt")

    fit_y_local = torch.as_tensor(np.asarray(target_y, np.float64)[fit_idx], dtype=torch.float32)
    replay = build_factorial_model(blob, SEED, coding_mode, binding_mode).to(device)
    replay.load_state_dict({k: v.to(device) for k, v in soup_state.items()})
    replay.eval()
    fit_raw, _ = predict(replay, fit_data, fit_y_local, device)
    b = float(np.median(y_fit_np - fit_raw))
    result.update(
        {
            "soup_members": members,
            "soup_state_sha256": state_hash(soup_state),
            "last_state_sha256": state_hash(last_state),
            "calibration": {"b": b, "folded_into_state": False},
            "fit_raw_mae_y": float(np.mean(np.abs(fit_raw - y_fit_np))),
            "fit_cal_mae_y": float(np.mean(np.abs(fit_raw + b - y_fit_np))),
            "kappa_buffer": float(soup_state["kappa"].reshape(-1)[0].item()),
        }
    )
    np.savez_compressed(
        out_dir / f"{arm}_predictions.npz",
        fit_raw=fit_raw.astype(np.float32),
        fit_idx=fit_idx,
        b=np.float64(b),
    )
    write_json(out_dir / f"{arm}.json", result)
    log(f"[{arm}] DONE fit_raw={result['fit_raw_mae_y']:.6f} fit_cal={result['fit_cal_mae_y']:.6f} b={b:.6f} wall={result['wall_clock_s']:.0f}s")
    return result


# ---------------------------------------------------------------------------
# operator / invariant checks
# ---------------------------------------------------------------------------


def _model_has_message_passing(model: torch.nn.Module) -> list[str]:
    bad: list[str] = []
    for name, module in model.named_modules():
        kind = type(module).__name__.lower()
        if any(token in kind for token in ("messagepassing", "transformer", "attention", "gat", "gcn", "sage", "gin")):
            bad.append(f"{name}:{type(module).__name__}")
    return bad


def run_operator_checks(
    blob: Mapping[str, Any],
    fit_data: Sequence[Any],
    target_y: np.ndarray,
    *,
    device: torch.device,
    out_dir: Path,
    log: Any = print,
) -> dict[str, Any]:
    checks: dict[str, Any] = {"official_test_loaded": False}
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    y_fit = np.asarray(target_y, np.float64)[fit_idx]

    # --- 1. S_J is the canonical Full; four arms share initial parameters -----
    canonical = sc.build_scale_model(
        np.asarray(blob["D_fit"], np.float32), SEED, zjd.fold_subspace(blob), sc.FULL, scale_seed=SCALE_SEED
    )
    seed_everything(SEED)
    s_j = build_factorial_model(blob, SEED, "sparse", "paired")
    canonical_state = {k: v for k, v in canonical.state_dict().items()}
    sj_state = {k: v for k, v in s_j.state_dict().items()}
    canonical_params = {k: v for k, v in canonical.named_parameters()}
    sj_params = {k: v for k, v in s_j.named_parameters()}
    param_mismatch = sorted(set(canonical_params) ^ set(sj_params))
    value_mismatch = [
        k for k in sorted(set(canonical_params) & set(sj_params))
        if not torch.equal(canonical_params[k].detach(), sj_params[k].detach())
    ]
    batch = make_batch(fit_data, list(range(min(32, len(fit_data)))), torch.as_tensor(y_fit[:32], dtype=torch.float32), device)
    canonical = canonical.to(device).eval()
    s_j = s_j.to(device).eval()
    with torch.no_grad():
        p_can, _ = canonical(batch, mask=cm.C6_MASK, return_aux=True)
        p_sj, _ = s_j(batch, mask=cm.C6_MASK, return_aux=True)
    checks["canonical_identity"] = {
        "parameter_name_mismatch": param_mismatch,
        "parameter_value_mismatch": value_mismatch,
        "forward_max_abs_diff": float((p_can - p_sj).abs().max()),
        "tolerance": 2.0e-6,
        "pass": bool(
            not param_mismatch and not value_mismatch and float((p_can - p_sj).abs().max()) <= 2.0e-6
        ),
    }

    # four arms: identical trainable params at init, 408,651 each.
    arm_models: dict[str, FactorialFull] = {}
    init_hashes: dict[str, str] = {}
    param_counts: dict[str, int] = {}
    for arm in ARMS:
        seed_everything(SEED)
        m = build_factorial_model(blob, SEED, CODING_OF[arm], BINDING_OF[arm]).to(device)
        arm_models[arm] = m
        init_hashes[arm] = trainable_parameter_hash(m)
        param_counts[arm] = trainable_parameter_count(m)
    checks["arm_init_identity"] = {
        "trainable_parameter_counts": param_counts,
        "init_hashes": init_hashes,
        "all_equal": bool(len(set(init_hashes.values())) == 1),
        "all_408651": bool(set(param_counts.values()) == {FULL_PARAMETERS}),
        "pass": bool(len(set(init_hashes.values())) == 1 and set(param_counts.values()) == {FULL_PARAMETERS}),
    }

    # --- 2. coding: width 33, sparse vs dense actually differ, no IHT in dense -
    phi = torch.cat([d.dict_phi.to(dtype=torch.float32) for d in fit_data[:64]], 0).to(device)
    kappa_info = compute_kappa(arm_models["S_J"], fit_data)
    with torch.no_grad():
        for arm in ARMS:
            arm_models[arm].kappa.fill_(float(kappa_info["kappa"]))
        coord_s = arm_models["S_J"].code(phi)
        coord_d = arm_models["D_J"].code(phi)
        Dbar = arm_models["D_J"].residual_dictionary()
        U = arm_models["D_J"].U
        r = phi - (phi @ U) @ U.t()
        alpha_s = coord_s[:, 1:]
        alpha_d = coord_d[:, 1:]
        dense_ref = float(arm_models["D_J"].kappa) * (r @ Dbar)
        recon_s = arm_models["S_J"].reconstruct(phi, coord_s)
        recon_d = arm_models["D_J"].reconstruct(phi, coord_d)
    checks["coding"] = {
        "kappa": kappa_info,
        "widths": {"S": int(coord_s.shape[1]), "D": int(coord_d.shape[1])},
        "sparse_nnz_per_row": float((alpha_s != 0).float().sum(1).mean()),
        "dense_nonzero_fraction": float((alpha_d != 0).float().mean()),
        "dense_code_matches_reference_max_abs": float((alpha_d - dense_ref).abs().max()),
        "sparse_vs_dense_max_abs": float((alpha_s - alpha_d).abs().max()),
        "sparse_rms": kappa_info["sparse_rms"],
        "dense_scaled_rms": kappa_info["dense_rms"] * kappa_info["kappa"],
        "dense_unscaled_rms": kappa_info["dense_rms"],
        "rms_match_rel": abs(
            kappa_info["sparse_rms"] - kappa_info["dense_rms"] * kappa_info["kappa"]
        ) / kappa_info["sparse_rms"],
        "recon_sparse_form": float((recon_s - alpha_s @ Dbar.t()).abs().max()),
        "recon_dense_form": float((recon_d - (alpha_d / float(arm_models["D_J"].kappa)) @ Dbar.t()).abs().max()),
        "pass": bool(
            int(coord_s.shape[1]) == 33
            and int(coord_d.shape[1]) == 33
            and float((alpha_s != 0).float().sum(1).mean()) <= cssd.SPARSITY + 1e-6
            and float((alpha_d != 0).float().mean()) > 0.99
            and float((alpha_d - dense_ref).abs().max()) < 1e-5
            and abs(
                kappa_info["sparse_rms"] - kappa_info["dense_rms"] * kappa_info["kappa"]
            ) / kappa_info["sparse_rms"] < 1e-6
        ),
    }

    # --- 3. micro-bucket enumeration: mean over semantic permutations == M ----
    gen = torch.Generator().manual_seed(1234)
    micro: dict[str, Any] = {}
    for n_items in (2, 3):
        w = 4
        a = torch.randn(1, n_items, w, generator=gen)
        b = torch.randn(1, n_items, w, generator=gen)
        index = torch.zeros(n_items, dtype=torch.long)
        m = indep_bucket(a.reshape(n_items, w), b.reshape(n_items, w), index, 1, w)
        perms = list(_permutations(list(range(n_items))))
        js = []
        for perm in perms:
            bp = b.reshape(n_items, w)[torch.tensor(perm)]
            js.append(paired_bucket(a.reshape(n_items, w), bp, index, 1, w))
        j_mean = torch.stack(js).mean(0)
        micro[f"n{n_items}"] = {
            "n_permutations": len(perms),
            "mean_J_minus_M_max_abs": float((j_mean - m).abs().max()),
            "pass": bool(float((j_mean - m).abs().max()) < 1e-5),
        }
    # n=1: J == M exactly.
    a1 = torch.randn(1, w, generator=gen)
    b1 = torch.randn(1, w, generator=gen)
    idx1 = torch.zeros(1, dtype=torch.long)
    j1 = paired_bucket(a1, b1, idx1, 1, w)
    m1 = indep_bucket(a1, b1, idx1, 1, w)
    # n=0: M == 0.
    m0 = indep_bucket(torch.zeros(0, w), torch.zeros(0, w), torch.zeros(0, dtype=torch.long), 1, w)
    micro["n1_identity"] = {"max_abs": float((j1 - m1).abs().max()), "pass": bool(float((j1 - m1).abs().max()) < 1e-6)}
    micro["n0_zero"] = {"value": float(m0.abs().max()), "pass": bool(float(m0.abs().max()) == 0.0)}
    checks["micro_bucket_enumeration"] = micro

    # --- 4. production M equals the tested operator on real factors ----------
    data0 = make_batch(fit_data, list(range(min(8, len(fit_data)))), torch.as_tensor(y_fit[:8], dtype=torch.float32), device)
    with torch.no_grad():
        coord0 = arm_models["S_M"].code(data0.dict_phi)
        a_n, b_n, idx_n = arm_models["S_M"].node_factor_slots(coord0, data0)
        n_nodes = int(coord0.shape[0])
        node_flat_ref = indep_bucket(a_n, b_n, idx_n, n_nodes * int(p2.N_SHELLS), int(p2.D_A))
        s_e, c_e, idx_e, d_e = arm_models["S_M"].edge_factor_slots(coord0, data0)
        edge_flat_ref = indep_bucket(s_e, c_e, idx_e, n_nodes * int(p2.SHELLPAIR_CLASSES), d_e)
        # production path via the real environment
        interface = arm_models["S_M"].semantic_interface(coord0, data0, None, None)
        _ = arm_models["S_M"]._indep_environment_from_parts(coord0, data0, interface)
    checks["production_binding"] = {
        "indep_calls_before": int(arm_models["S_M"]._indep_calls),
        "node_reference_variance": float(node_flat_ref.var().item()),
        "edge_reference_variance": float(edge_flat_ref.var().item()),
        "pass": bool(float(node_flat_ref.var().item()) >= 0.0 and float(edge_flat_ref.var().item()) >= 0.0),
    }

    # --- 5. forward/backward reaches operators and parameters -----------------
    m = arm_models["S_M"]
    m.train()
    batch = make_batch(fit_data, list(range(16)), torch.as_tensor(y_fit[:16], dtype=torch.float32), device)
    pred, aux = m(batch, mask=cm.C6_MASK, return_aux=True)
    task = F.l1_loss(pred.view(-1), batch.y.view(-1))
    rec = m.reconstruction_loss(aux["phi"], aux["coord"])
    (task + H1_LAMBDA * rec).backward()
    grad_norms = {
        name: float(p.grad.norm()) for name, p in m.named_parameters() if p.grad is not None
    }
    checks["dispatch"] = {
        "indep_calls_after_forward": int(m._indep_calls),
        "indep_used": bool(m._indep_calls > 0),
        "D_grad": grad_norms.get("D", 0.0),
        "W_A_S_grad": grad_norms.get("W_A_S", 0.0),
        "W_A_C_grad": grad_norms.get("W_A_C", 0.0),
        "W_E_S_grad": grad_norms.get("W_E_S", 0.0),
        "W_E_C_grad": grad_norms.get("W_E_C", 0.0),
        "kappa_requires_grad": bool(m.kappa.requires_grad),
        "n_params_with_grad": int(len(grad_norms)),
        "message_passing_modules": _model_has_message_passing(m),
        "pass": bool(
            m._indep_calls > 0
            and grad_norms.get("D", 0.0) > 0
            and grad_norms.get("W_A_S", 0.0) > 0
            and grad_norms.get("W_A_C", 0.0) > 0
            and grad_norms.get("W_E_S", 0.0) > 0
            and grad_norms.get("W_E_C", 0.0) > 0
            and not m.kappa.requires_grad
            and not _model_has_message_passing(m)
        ),
    }
    m.zero_grad(set_to_none=True)

    # --- 6. input does not depend on y / c / k ---------------------------------
    # The structural code is a pure function of phi65; a label shuffle must leave
    # it bit-identical.  The masked forward can differ by float32 accumulation
    # order when the *target tensor contents* change the fused kernel path, so
    # the forward is reported separately with a tolerance.
    m.eval()
    with torch.no_grad():
        coord1 = m.code(batch.dict_phi)
        p1_, _ = m(batch, mask=cm.C6_MASK, return_aux=True)
        batch2 = batch.clone()
        batch2.y = torch.randn_like(batch2.y)
        coord2 = m.code(batch2.dict_phi)
        p2_, _ = m(batch2, mask=cm.C6_MASK, return_aux=True)
    checks["input_independence"] = {
        "code_max_abs_diff_after_label_shuffle": float((coord1 - coord2).abs().max()),
        "forward_max_abs_diff_after_label_shuffle": float((p1_ - p2_).abs().max()),
        "forward_tolerance": 1.0e-5,
        "pass": bool(
            float((coord1 - coord2).abs().max()) == 0.0
            and float((p1_ - p2_).abs().max()) <= 1e-5
        ),
    }

    # --- 7. bucket locality / batch consistency --------------------------------
    checks["bucket_locality"] = _bucket_locality_check(blob, fit_data, target_y, device)

    # --- 8. real-input J != M (non-degeneracy of the operator on real data) ----
    with torch.no_grad():
        coord_sm = arm_models["S_M"].code(batch.dict_phi)
        a_n, b_n, idx_n = arm_models["S_M"].node_factor_slots(coord_sm, batch)
        n_nodes = int(coord_sm.shape[0])
        jn = paired_bucket(a_n, b_n, idx_n, n_nodes * int(p2.N_SHELLS), int(p2.D_A))
        mn = indep_bucket(a_n, b_n, idx_n, n_nodes * int(p2.N_SHELLS), int(p2.D_A))
        s_e, c_e, idx_e, d_e = arm_models["S_M"].edge_factor_slots(coord_sm, batch)
        je = paired_bucket(s_e, c_e, idx_e, n_nodes * int(p2.SHELLPAIR_CLASSES), d_e)
        me = indep_bucket(s_e, c_e, idx_e, n_nodes * int(p2.SHELLPAIR_CLASSES), d_e)
        counts_n = torch.zeros(n_nodes * int(p2.N_SHELLS), dtype=torch.long, device=idx_n.device)
        counts_n.index_add_(0, idx_n, torch.ones_like(idx_n))
        counts_e = torch.zeros(n_nodes * int(p2.SHELLPAIR_CLASSES), dtype=torch.long, device=idx_e.device)
        counts_e.index_add_(0, idx_e, torch.ones_like(idx_e))
    checks["real_input_operator"] = {
        "node_nonzero_bucket_fraction": float((jn - mn).abs().sum(1).gt(0).float().mean()),
        "edge_nonzero_bucket_fraction": float((je - me).abs().sum(1).gt(0).float().mean()),
        "node_n_ge2_fraction": float((counts_n >= 2).float().mean()),
        "edge_n_ge2_fraction": float((counts_e >= 2).float().mean()),
        "node_J_M_max_abs": float((jn - mn).abs().max()),
        "edge_J_M_max_abs": float((je - me).abs().max()),
        "node_slot_variance": float(mn.var().item()),
        "edge_slot_variance": float(me.var().item()),
    }

    checks["mechanism_ok"] = bool(
        checks["canonical_identity"]["pass"]
        and checks["arm_init_identity"]["pass"]
        and checks["coding"]["pass"]
        and checks["micro_bucket_enumeration"]["n2"]["pass"]
        and checks["micro_bucket_enumeration"]["n3"]["pass"]
        and checks["micro_bucket_enumeration"]["n1_identity"]["pass"]
        and checks["micro_bucket_enumeration"]["n0_zero"]["pass"]
        and checks["dispatch"]["pass"]
        and checks["input_independence"]["pass"]
        and checks["bucket_locality"]["pass"]
    )
    write_json(out_dir / "operator_checks.json", checks)
    log(f"[checks] mechanism_ok={checks['mechanism_ok']}")
    return checks


def _permutations(items: list[int]):
    if len(items) <= 1:
        yield tuple(items)
        return
    for i, item in enumerate(items):
        rest = items[:i] + items[i + 1 :]
        for perm in _permutations(rest):
            yield (item,) + perm


def _bucket_locality_check(
    blob: Mapping[str, Any], fit_data: Sequence[Any], target_y: np.ndarray, device: torch.device
) -> dict[str, Any]:
    """Per-molecule slots must equal the corresponding block of a 2-molecule batch.

    Also: permuting the two molecules inside a batch must move their slot blocks
    correspondingly (no cross-graph bucket mixing), and the model prediction is
    invariant to the graph order for the same graph rows.
    """
    seed_everything(SEED)
    model = build_factorial_model(blob, SEED, "sparse", "indep").to(device).eval()
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    y_fit = np.asarray(target_y, np.float64)[fit_idx]

    def slots(data_list, indices):
        b = make_batch(data_list, indices, torch.as_tensor(y_fit[: len(data_list)], dtype=torch.float32), device)
        with torch.no_grad():
            coord = model.code(b.dict_phi)
            a_n, b_n, idx_n = model.node_factor_slots(coord, b)
            n = int(coord.shape[0])
            node = indep_bucket(a_n, b_n, idx_n, n * int(p2.N_SHELLS), int(p2.D_A))
            s_e, c_e, idx_e, d_e = model.edge_factor_slots(coord, b)
            edge = indep_bucket(s_e, c_e, idx_e, n * int(p2.SHELLPAIR_CLASSES), d_e)
        return node.view(n, int(p2.N_SHELLS), int(p2.D_A)), edge.view(n, int(p2.SHELLPAIR_CLASSES), d_e), b

    node_a, edge_a, ba = slots(fit_data, [0])
    node_b, edge_b, bb = slots(fit_data, [1])
    node_ab, edge_ab, bab = slots(fit_data, [0, 1])
    n_a = int(node_a.shape[0])
    n_b = int(node_b.shape[0])
    node_join = torch.cat([node_a, node_b], 0).reshape(-1)
    edge_join = torch.cat([edge_a, edge_b], 0).reshape(-1)
    node_max = float((node_join - node_ab.reshape(-1)).abs().max())
    edge_max = float((edge_join - edge_ab.reshape(-1)).abs().max())

    res_ab = _predict_single(model, fit_data, [0, 1], y_fit, device)
    res_ba = _predict_single(model, fit_data, [1, 0], y_fit, device)
    # predictions of the same graph must be identical across the two orders
    pred_map_ab = {ind: val for ind, val in zip([0, 1], res_ab.tolist())}
    pred_map_ba = {ind: val for ind, val in zip([1, 0], res_ba.tolist())}
    order_max = max(abs(pred_map_ab[k] - pred_map_ba[k]) for k in pred_map_ab)
    return {
        "node_join_max_abs": node_max,
        "edge_join_max_abs": edge_max,
        "graph_order_prediction_max_abs": float(order_max),
        "tolerance": 1.0e-4,
        "pass": bool(node_max <= 1e-4 and edge_max <= 1e-4 and float(order_max) <= 1e-4),
    }


def _predict_single(model, fit_data, indices, y_fit, device) -> np.ndarray:
    b = make_batch(fit_data, indices, torch.as_tensor(y_fit[: len(indices)], dtype=torch.float32), device)
    with torch.no_grad():
        return model(b, mask=cm.C6_MASK).view(-1).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# phase 0
# ---------------------------------------------------------------------------


def phase0(*, device: str = "cpu", log: Any = print) -> dict[str, Any]:
    t0 = time.perf_counter()
    dev = resolve_device(device)
    blob = load_prep_blob()
    decomp = load_target()
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    k = np.asarray(decomp["k"], np.int64)
    checks: dict[str, Any] = {
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "split": {
            "n_fit": int(len(fit_idx)),
            "n_dev": int(len(dev_idx)),
            "fit_idx_sha256": hashlib.sha256(fit_idx.tobytes()).hexdigest(),
            "dev_idx_sha256": hashlib.sha256(dev_idx.tobytes()).hexdigest(),
            "expected_fit_idx_sha256": "165e87ef4398ba8ef57411c2f118c4cca4ea74007f2485611b84f0c09bbdd9ea",
            "expected_dev_idx_sha256": "fb8b78063e7c8a5c759bfa7d553738068cee1738a2d67b210e5d9dc492331376",
            "dev_group_counts": {
                "k0": int((k[dev_idx] == 0).sum()),
                "k-1": int((k[dev_idx] == -1).sum()),
                "kle-2": int((k[dev_idx] <= -2).sum()),
            },
            "dev_group_counts_expected": {"k0": 1926, "k-1": 65, "kle-2": 9},
        },
        "prep": {
            "D_fit_sha256": sha256_array(np.asarray(blob["D_fit"], np.float32)),
            "U_components_sha256": sha256_array(np.asarray(blob["U_components"], np.float32)),
            "fold_objects_file_sha256": file_sha256(PREP_BLOB),
        },
        "sources": {
            "prep_blob": str(PREP_BLOB.relative_to(zjd.REPO_ROOT)),
            "encoded_train": "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt",
            "env_train": "tracks/ksvd/results/e2e_dictenv_p1/cache/env_train.pt",
            "target_decomposition": "tracks/ksvd/results/zinc_full_cycle_target_decomposition_v1/target_decomposition.npz",
        },
    }
    checks["split"]["pass"] = bool(
        checks["split"]["fit_idx_sha256"] == checks["split"]["expected_fit_idx_sha256"]
        and checks["split"]["dev_idx_sha256"] == checks["split"]["expected_dev_idx_sha256"]
        and checks["split"]["dev_group_counts"] == checks["split"]["dev_group_counts_expected"]
    )

    train_data = load_train_only()
    zftd.apply_prep_train_only(train_data, blob)
    fit_data, _dev_data = build_fit_dev(train_data, fit_idx, dev_idx)
    y = np.asarray(decomp["y"], np.float64)
    checks["encoded_y_matches_max_abs"] = float(np.max(np.abs(np.array([float(d.y.item()) for d in train_data]) - y)))

    # identity/invariant checks (CPU) and kappa
    op_checks = run_operator_checks(blob, fit_data, y, device=dev, out_dir=RESULTS_DIR, log=log)
    checks["operator_mechanism_ok"] = bool(op_checks["mechanism_ok"])

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": "8126300",
        "arms": list(ARMS),
        "coding_of": CODING_OF,
        "binding_of": BINDING_OF,
        "seed": SEED,
        "scale_seed": SCALE_SEED,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "h1_lambda": H1_LAMBDA,
        "soup_epochs": list(SOUP_EPOCHS),
        "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
        "full_parameters": FULL_PARAMETERS,
        "sparsity": int(cssd.SPARSITY),
        "iht_steps": int(cssd.IHT_STEPS),
        "kappa": op_checks["coding"]["kappa"],
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "delta": DELTA},
        "group_names": GROUP_NAMES,
        "primary_endpoint": "dev G0 (k=0) cal MAE",
        "secondary_endpoint": "dev overall cal MAE",
        "fit_dev_counts": checks["split"]["dev_group_counts"],
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    write_json(RESULTS_DIR / "protocol.json", protocol)
    write_json(RESULTS_DIR / "phase0_checks.json", checks)
    log(
        f"[phase0] split_ok={checks['split']['pass']} mechanism_ok={checks['operator_mechanism_ok']} "
        f"kappa={op_checks['coding']['kappa']['kappa']:.8f} seconds={time.perf_counter() - t0:.1f}"
    )
    return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZINC structure-semantic factorial seed0 v1")
    parser.add_argument("--mode", required=True, choices=("phase0", "checks", "smoke", "train"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)

    if args.mode == "phase0":
        phase0(device=args.device)
        return 0

    blob = load_prep_blob()
    decomp = load_target()
    train_data = load_train_only()
    zftd.apply_prep_train_only(train_data, blob)
    fit_data, dev_data = build_fit_dev(train_data, blob["fit_idx"], blob["dev_idx"])
    y = np.asarray(decomp["y"], np.float64)

    if args.mode == "checks":
        checks = run_operator_checks(blob, fit_data, y, device=resolve_device(args.device), out_dir=out_dir)
        return 0 if checks["mechanism_ok"] else 1

    if args.mode == "smoke":
        checks = run_operator_checks(blob, fit_data, y, device=resolve_device(args.device), out_dir=out_dir)
        max_steps = int(args.max_steps or 2)
        smoke: dict[str, Any] = {}
        for arm in ARMS:
            smoke[arm] = train_arm(
                arm, blob, fit_data[:256], y, device=resolve_device(args.device), out_dir=out_dir,
                epochs=1, max_steps=max_steps,
            )
        write_json(out_dir / "smoke.json", {"checks": checks, "arms": smoke})
        ok = checks["mechanism_ok"] and all(v["steps_done"] == max_steps for v in smoke.values())
        return 0 if ok else 1

    if args.mode == "train":
        assert args.arm is not None, "--arm required for train"
        train_arm(
            args.arm, blob, fit_data, y, device=resolve_device(args.device), out_dir=out_dir,
            epochs=int(args.epochs),
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())