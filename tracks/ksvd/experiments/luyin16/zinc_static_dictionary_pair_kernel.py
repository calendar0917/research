"""SDPK-v0 -- Static Dictionary-Conditioned Pair Kernel (strict-static ZINC).

Round: **ZINC-static-dictionary-pair-kernel-v0**.

Single performance-oriented question
------------------------------------
If the learnable dictionary is promoted from a *bypassable local residual
adapter* (previous round: ``h_i = h0_i + gamma * delta_dict``) to the
**core coordinate system of the occurrence-level static pair / relation
kernel**, does absolute strict-static ZINC MAE open a new performance band?

Strict-static contract (inherited, non-negotiable)
--------------------------------------------------
::

    NO message passing
    NO pair -> centre/patch write-back
    NO relation refresh
    NO second pair evaluation

The local state is exactly the S0 local state::

    x_i -> patch_encoder -> h_i in R^48          # h_i = h0_i, no residual
    s_i = l2_normalize(Wq(h_i))                  # 48 -> 32
    D_k = l2_normalize(dictionary_k)             # 64 x 32
    tau = 0.05 + 0.95 * sigmoid(tau_logit)
    alpha_i = softmax((s_i @ D^T) / tau)         # 64D, end-to-end
    c_i = U(alpha_i)                             # 64 -> 16

The dictionary coordinates enter **one** occurrence-level pair kernel::

    dict_sum  = c_i + c_j
    dict_diff = |c_i - c_j|
    dict_prod = c_i * c_j
    m_ij      = 1 + tanh(Wm([dict_sum, dict_diff, dict_prod, rel_ij]))
    conditioned_prod = (u_i * u_j) * distance_gate * m_ij
    pair_input = [u_i+u_j, |u_i-u_j|, conditioned_prod, rel_ij,
                  dict_sum, dict_diff, dict_prod]        # 112D
    q_ij = pair_encoder(pair_input)                     # 16D, computed ONCE

``q_ij`` never updates ``h_i`` / ``h_j``.  Everything is trained end-to-end
against the inherited L1/MAE objective.  No K-SVD / reconstruction / sparsity /
entropy / balance / orthogonality / top-k term exists.

Frozen hyperparameters (no sweep, no rescue)
--------------------------------------------
``K=64``, ``dict_rank=32``, ``coord_dim=16``, ``tau_init=0.20``,
pair encoder ``112 -> 64 -> 16``, head ``302 -> 13 -> 13 -> 1`` (inherited S0).

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair_kernel <stage>

Stages: ``param_audit integrity encode baseline_guard smoke train analyze all``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/zinc_static_dictionary_pair_kernel_v0"
RUNS_DIR = RESULTS_DIR / "runs"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"

EXPERIMENT_NAME = "zinc_static_dictionary_pair_kernel_v0"
PROTOCOL_VERSION = "zinc_static_dictionary_pair_kernel_v0"

TYPED_VOCAB_SIZE = sdp.TYPED_VOCAB_SIZE
PARENT_VOCAB_SIZE = sdp.PARENT_VOCAB_SIZE
HEAD_HIDDEN = sdp.HEAD_HIDDEN  # (13, 13) -- inherited S0 small raw head

# --- dictionary (fixed; no sweep) ------------------------------------------
DICT_ATOMS = 64
DICT_RANK = 32
COORD_DIM = 16
DICT_TAU_INIT = 0.20

# --- training protocol (inherited optimized strict-static regime) ----------
OPTIMIZED_PROTOCOL: dict[str, Any] = dict(sdp.OPTIMIZED_PROTOCOL)
OPTIMIZED_PROTOCOL["early_termination"] = "none (full 240 epochs)"

# --- pre-registered seed-0 performance gates (Top-5 soup is primary) -------
S0_SEED0_BEST = 0.145674
S0_SEED0_SOUP = 0.140794
S0_SEED1_BEST = 0.139389
S0_SEED1_SOUP = 0.136423

GATE_SEED0_SOUP = 0.1328       # absolute soup <= this -> GO
GATE_SEED0_SOUP_DELTA = 0.008  # improvement over S0 seed0 soup
GATE_SEED0_BEST_DELTA = 0.006  # best-checkpoint improvement over S0 seed0
GATE_SEED1_SOUP = 0.1300       # absolute soup <= this -> strong confirmation
GATE_SEED1_SOUP_DELTA = 0.006

MAX_FULL_RUNS = 2


# ---------------------------------------------------------------------------
# small io helpers (kept local; the durable notes are written locally)
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    sdp._write_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    return sdp._read_json(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    sdp._write_csv(path, rows)


def _n_params(module: nn.Module | None) -> int:
    return sdp._n_params(module)


def _git_commit() -> str:
    return sdp._git_commit()


def _environment_fingerprint(device: str = "cpu") -> dict[str, Any]:
    return sdp._environment_fingerprint(device)


def _configure_determinism() -> None:
    sdp._configure_determinism()


def _seed_everything(seed: int) -> None:
    sdp._seed_everything(seed)


def _tau_from_logit(tau_logit: torch.Tensor) -> torch.Tensor:
    return 0.05 + 0.95 * torch.sigmoid(tau_logit)


def _tau_logit_for(tau: float) -> float:
    return sdp._tau_logit_for(tau)


# ---------------------------------------------------------------------------
# dictionary coordinates
# ---------------------------------------------------------------------------


class DictionaryCoordinates(nn.Module):
    """End-to-end learnable dictionary coordinates over the local state.

    ``h -> alpha -> c``.  It is a pure function of the local patch state and
    never sees pair information, so it can neither perform message passing nor
    be refreshed by a pair result.
    """

    def __init__(
        self,
        hidden: int,
        rank: int = DICT_RANK,
        atoms: int = DICT_ATOMS,
        coord_dim: int = COORD_DIM,
        tau_init: float = DICT_TAU_INIT,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.atoms_count = int(atoms)
        self.coord_dim = int(coord_dim)
        self.query = nn.Linear(self.hidden, self.rank)
        self.atoms = nn.Parameter(torch.empty(self.atoms_count, self.rank))
        self.coord_proj = nn.Linear(self.atoms_count, self.coord_dim)
        self.tau_logit = nn.Parameter(torch.tensor(_tau_logit_for(tau_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.query.reset_parameters()
        self.coord_proj.reset_parameters()
        nn.init.normal_(self.atoms, std=1.0 / math.sqrt(self.rank))

    def current_tau(self) -> torch.Tensor:
        return _tau_from_logit(self.tau_logit)

    def normalized_atoms(self) -> torch.Tensor:
        return F.normalize(self.atoms, dim=-1, eps=1.0e-8)

    def alpha_from_h(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = F.normalize(self.query(h), dim=-1, eps=1.0e-8)
        atoms = self.normalized_atoms()
        tau = self.current_tau()
        alpha = F.softmax((query @ atoms.t()) / tau, dim=-1)
        return alpha, tau, atoms

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        alpha, _tau, _atoms = self.alpha_from_h(h)
        return self.coord_proj(alpha)


# ---------------------------------------------------------------------------
# SDPK model
# ---------------------------------------------------------------------------


class SDPKModel(zpp.PatchPathModel):
    """Strict-static patch-pair backbone with a dictionary-conditioned pair kernel.

    The pair kernel consumes the dictionary coordinates and gates the endpoint
    multiplicative interaction.  It is evaluated exactly once per pair and
    never writes back into any local state.
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        dict_rank: int = DICT_RANK,
        dict_atoms: int = DICT_ATOMS,
        coord_dim: int = COORD_DIM,
        dict_tau_init: float = DICT_TAU_INIT,
        head_hidden: Sequence[int] = HEAD_HIDDEN,
        **kwargs: Any,
    ) -> None:
        kwargs = dict(kwargs)
        kwargs["center_context"] = False
        dropout = float(kwargs.get("dropout", 0.05))
        super().__init__(
            int(typed_vocabulary_size), int(parent_vocabulary_size), **kwargs
        )
        if self.center_update is not None:  # pragma: no cover - contract guard
            raise RuntimeError("SDPK strict-static model must have center_update=None")

        self.dictionary = DictionaryCoordinates(
            int(self.patch_hidden),
            int(dict_rank),
            int(dict_atoms),
            int(coord_dim),
            float(dict_tau_init),
        )
        pair_hidden = int(self.pair_hidden)
        self.coord_dim = int(coord_dim)
        # Dictionary-conditioned multiplicative gate: [dict_sum|diff|prod, rel].
        self.dict_gate = nn.Linear(3 * int(coord_dim) + pair_hidden, pair_hidden)
        # One static pair kernel over the widened (112D) pair input.
        self.pair_encoder = zpp._MLPBlock(
            3 * pair_hidden + pair_hidden + 3 * int(coord_dim),
            max(2 * pair_hidden, 64),
            pair_hidden,
            dropout,
        )
        self.head = GenericReader(
            int(self.unified_graph_width), tuple(int(h) for h in head_hidden)
        )
        self.dict_intervention: str | None = None

    # -- diagnostics hooks ---------------------------------------------------
    def set_dict_intervention(self, mode: str | None) -> "SDPKModel":
        if mode not in (None, "mean_coord", "neutral"):
            raise ValueError(f"unknown dictionary intervention {mode!r}")
        self.dict_intervention = mode
        return self

    @staticmethod
    def _graph_mean(
        value: torch.Tensor, batch: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        total = torch.zeros(
            (int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype
        )
        counts = torch.zeros(
            (int(n_graphs), 1), device=value.device, dtype=value.dtype
        )
        total.index_add_(0, batch, value)
        counts.index_add_(
            0, batch, torch.ones((batch.shape[0], 1), device=value.device, dtype=value.dtype)
        )
        return total / counts.clamp_min(1.0)

    # -- forward -------------------------------------------------------------
    def encode(self, data: Data) -> torch.Tensor:
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])
        e_patch = self._patch_token_value(data)
        structural_blocks: list[torch.Tensor] = []
        if self.structural_context_mode != "none":
            e_ctx = self._structural_context_embedding_value(data)
            if self.structural_context_fusion == "condition":
                e_patch = self._condition_patch(
                    e_patch, e_ctx, self._structural_no_ring_mask(data)
                )
            else:
                structural_blocks.append(e_ctx)
        attribute_blocks: list[torch.Tensor] = []
        if self.attribute_encoder is not None:
            attribute_blocks.append(self.attribute_encoder(data, e_patch))
        # h_i = LocalEncoder(x_i).  No dictionary residual, no gamma.
        patch = self.patch_encoder(
            torch.cat(
                [
                    data.patch_cont,
                    data.patch_context,
                    e_patch,
                    self.parent_embedding(data.parent_token),
                    *structural_blocks,
                    *attribute_blocks,
                ],
                dim=1,
            )
        )
        unary = self._pool_nodes(patch, data.batch, n_graphs)
        direct_blocks: list[torch.Tensor] = []
        if self.direct_token_readout:
            token_code = self.typed_embedding.embedding(data.typed_token)
            direct_blocks.append(
                self._pool_values(token_code, data.batch, n_graphs, "moments")
            )

        # --- dictionary coordinates (pure function of the local state) ------
        coord = self.dictionary(patch)
        if self.dict_intervention == "mean_coord":
            coord = self._graph_mean(coord, data.batch, n_graphs)[data.batch]
        elif self.dict_intervention == "neutral":
            coord = torch.zeros_like(coord)

        source = data.pair_index[0]
        target = data.pair_index[1]
        projected_left = self.pair_projection(patch[source])
        projected_right = self.pair_projection(patch[target])
        relation = self.relation_encoder(data.pair_relation)
        product = projected_left * projected_right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))

        left = coord[source]
        right = coord[target]
        dict_sum = left + right
        dict_diff = torch.abs(left - right)
        dict_prod = left * right
        if self.dict_intervention == "neutral":
            modulation = torch.ones_like(product)
        else:
            gate_input = torch.cat([dict_sum, dict_diff, dict_prod, relation], dim=1)
            modulation = 1.0 + torch.tanh(self.dict_gate(gate_input))
        conditioned_prod = product * gate * modulation

        pair_input = torch.cat(
            [
                projected_left + projected_right,
                torch.abs(projected_left - projected_right),
                conditioned_prod,
                relation,
                dict_sum,
                dict_diff,
                dict_prod,
            ],
            dim=1,
        )
        # ONE static pair kernel; the result never re-enters the local state.
        pair_value = self.pair_encoder(pair_input)
        pair_batch = data.batch[source]
        relation_readout = self._pool_pairs(
            pair_value, pair_batch, data.pair_bucket, n_graphs
        )
        graph_hidden = self.global_encoder(global_context)
        readout_blocks = [unary, relation_readout, *direct_blocks, graph_hidden]
        if self.topology_encoder is not None:
            topology = data.topology_features
            if topology.ndim == 1:
                topology = topology.unsqueeze(0)
            if int(topology.shape[1]) != self.topology_input_width:
                raise RuntimeError(
                    f"topology width mismatch: data={int(topology.shape[1])} "
                    f"model={self.topology_input_width}"
                )
            readout_blocks.append(self.topology_encoder(topology))
        return torch.cat(readout_blocks, dim=1)


# ---------------------------------------------------------------------------
# construction with S0-matched shared tensors
# ---------------------------------------------------------------------------


def build_sdpl(seed: int = 0, **kwargs: Any) -> SDPKModel:
    """Build SDPK with every shared tensor bit-identical to the S0 reference."""
    _seed_everything(int(seed))
    reference = sdp.build_s0(int(seed))
    reference_state = {
        key: value.detach().clone() for key, value in reference.state_dict().items()
    }
    _seed_everything(int(seed))
    model = SDPKModel(
        TYPED_VOCAB_SIZE, PARENT_VOCAB_SIZE, **sdp.base_model_kwargs(), **kwargs
    )
    with torch.no_grad():
        state = model.state_dict()
        copied = 0
        for key, value in reference_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)
                copied += 1
    model.shared_tensors_copied = int(copied)  # type: ignore[attr-defined]
    return model


# Backwards-friendly alias.
build_model = build_sdpl


# ---------------------------------------------------------------------------
# parameter / initialization audit
# ---------------------------------------------------------------------------


def parameter_audit() -> dict[str, Any]:
    s0 = sdp.build_s0(0)
    model = build_sdpl(0)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "center_context": bool(model.center_context),
        "center_update_is_none": bool(model.center_update is None),
        "patch_hidden": int(model.patch_hidden),
        "pair_hidden": int(model.pair_hidden),
        "coord_dim": int(model.coord_dim),
        "unified_graph_width": int(model.unified_graph_width),
        "pair_input_width": int(model.pair_encoder.layers[0].in_features),
        "dict_gate_input_width": int(model.dict_gate.in_features),
        "head_hidden": list(HEAD_HIDDEN),
        "dictionary_spec": {
            "atoms": DICT_ATOMS,
            "rank": DICT_RANK,
            "coord_dim": COORD_DIM,
            "tau_init": DICT_TAU_INIT,
        },
        "params": {
            "s0_total": _n_params(s0),
            "s0_head": _n_params(s0.head),
            "sdpl_total": _n_params(model),
            "sdpl_head": _n_params(model.head),
            "sdpl_dictionary": _n_params(model.dictionary),
            "sdpl_dict_gate": _n_params(model.dict_gate),
            "sdpl_pair_encoder": _n_params(model.pair_encoder),
        },
        "target_budget": {
            "preferred_max": 82000,
            "hard_max": 90000,
            "within_preferred": bool(_n_params(model) <= 82000),
            "within_hard": bool(_n_params(model) < 90000),
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


def initialization_match() -> dict[str, Any]:
    s0 = sdp.build_s0(0)
    model = build_sdpl(0)
    s0_state = s0.state_dict()
    model_state = model.state_dict()
    new_prefixes = (
        "dictionary.",
        "dict_gate.",
        "patch_encoder.layers.",
    )
    shared_keys = sorted(
        key
        for key in s0_state
        if key in model_state
        and s0_state[key].shape == model_state[key].shape
        and not key.startswith(new_prefixes)
    )
    diffs = {
        key: float((s0_state[key] - model_state[key]).abs().max().item())
        for key in shared_keys
    }
    # The pair encoder is a genuinely widened module and is deliberately NOT
    # expected to match; verify explicitly that its input width grew.
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_shared_tensors": int(len(shared_keys)),
        "shared_keys": shared_keys,
        "max_abs_diff_s0_vs_sdpl": max(diffs.values()) if diffs else 0.0,
        "bit_identical_shared": bool(all(value == 0.0 for value in diffs.values())),
        "s0_pair_input_width": int(s0.pair_encoder.layers[0].in_features),
        "sdpl_pair_input_width": int(model.pair_encoder.layers[0].in_features),
        "pair_encoder_widened": bool(
            int(model.pair_encoder.layers[0].in_features)
            > int(s0.pair_encoder.layers[0].in_features)
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "initialization_match.json", payload)
    return payload


# ---------------------------------------------------------------------------
# strict-static integrity + gradient viability
# ---------------------------------------------------------------------------


def _contract_violation(*_args: Any, **_kwargs: Any) -> torch.Tensor:
    raise RuntimeError(
        "strict-static contract violated: _pool_pairs_to_centres was called"
    )


def _first_batch(data: Sequence[Any], device: torch.device, limit: int = 128):
    return sdp._first_batch(data, device, limit)


def static_contract_checks(model: SDPKModel, batch: Any) -> dict[str, Any]:
    model.eval()
    results: dict[str, Any] = {}

    results["center_context_false"] = bool(model.center_context is False)
    results["center_update_is_none"] = bool(model.center_update is None)
    if not results["center_context_false"] or not results["center_update_is_none"]:
        raise RuntimeError("strict-static contract failed: centre update present")

    # (A) monkeypatch _pool_pairs_to_centres to raise; forward must succeed.
    original = zpp.PatchPathModel._pool_pairs_to_centres
    zpp.PatchPathModel._pool_pairs_to_centres = _contract_violation
    try:
        with torch.no_grad():
            _ = model(batch)
        results["forward_without_pair_to_center"] = True
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original

    # (C) pair kernel evaluated exactly once; relation encoder exactly once.
    counters = {"pair_encoder": 0, "relation_encoder": 0, "dictionary": 0}
    handles = [
        model.pair_encoder.register_forward_hook(
            lambda *_a, _c=counters: _c.__setitem__("pair_encoder", _c["pair_encoder"] + 1)
        ),
        model.relation_encoder.register_forward_hook(
            lambda *_a, _c=counters: _c.__setitem__(
                "relation_encoder", _c["relation_encoder"] + 1
            )
        ),
        model.dictionary.register_forward_hook(
            lambda *_a, _c=counters: _c.__setitem__("dictionary", _c["dictionary"] + 1)
        ),
    ]
    try:
        with torch.no_grad():
            _ = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    results["pair_encoder_calls_per_forward"] = int(counters["pair_encoder"])
    results["relation_encoder_calls_per_forward"] = int(counters["relation_encoder"])
    results["dictionary_calls_per_forward"] = int(counters["dictionary"])
    if counters["pair_encoder"] != 1:
        raise RuntimeError("pair kernel was evaluated more than once per forward")
    if counters["dictionary"] != 1:
        raise RuntimeError("dictionary coordinates were evaluated more than once")

    # (B) no pair -> local feedback: h_i / alpha_i / c_i invariance.
    captured: dict[str, torch.Tensor] = {}

    def _capture_h(_module, _inputs, output):
        captured["h"] = output.detach().clone()

    handle_h = model.patch_encoder.register_forward_hook(_capture_h)
    try:
        with torch.no_grad():
            pred_before = model(batch)
        h_before = captured["h"].clone()
        alpha_before, _, _ = model.dictionary.alpha_from_h(h_before)
        coord_before = model.dictionary.coord_proj(alpha_before)

        mutated = batch.clone()
        if int(mutated.pair_relation.shape[0]) > 0:
            generator = torch.Generator(device="cpu").manual_seed(1234)
            mutated.pair_relation = torch.randn(
                mutated.pair_relation.shape, generator=generator
            ).to(mutated.pair_relation.device)
        with torch.no_grad():
            pred_after = model(mutated)
        h_after = captured["h"].clone()
        alpha_after, _, _ = model.dictionary.alpha_from_h(h_after)
        coord_after = model.dictionary.coord_proj(alpha_after)
    finally:
        handle_h.remove()

    results["h_identical_under_relation_mutation"] = bool(torch.equal(h_before, h_after))
    results["alpha_identical_under_relation_mutation"] = bool(
        torch.equal(alpha_before, alpha_after)
    )
    results["coord_identical_under_relation_mutation"] = bool(
        torch.equal(coord_before, coord_after)
    )
    results["max_abs_h_diff"] = float((h_before - h_after).abs().max().item())
    results["max_abs_alpha_diff"] = float((alpha_before - alpha_after).abs().max().item())
    results["max_abs_coord_diff"] = float((coord_before - coord_after).abs().max().item())
    results["prediction_changes_under_relation_mutation"] = bool(
        float((pred_before - pred_after).abs().max().item()) > 1.0e-8
    )
    if not results["h_identical_under_relation_mutation"]:
        raise RuntimeError("strict-static contract violated: pair feeds back into h_i")
    if not results["alpha_identical_under_relation_mutation"]:
        raise RuntimeError("dictionary coordinates changed under pair mutation")
    if not results["prediction_changes_under_relation_mutation"]:
        raise RuntimeError("relation mutation test is vacuous (prediction unchanged)")

    # Pair ordering permutation invariance.
    if int(batch.pair_index.shape[1]) > 1:
        permutation = torch.randperm(
            int(batch.pair_index.shape[1]), generator=torch.Generator().manual_seed(7)
        ).to(batch.pair_index.device)
        permuted = batch.clone()
        permuted.pair_index = batch.pair_index[:, permutation]
        permuted.pair_relation = batch.pair_relation[permutation]
        permuted.pair_bucket = batch.pair_bucket[permutation]
        with torch.no_grad():
            pred_permuted = model(permuted)
        results["pair_order_max_abs_pred_diff"] = float(
            (pred_before - pred_permuted).abs().max().item()
        )
        results["pair_order_invariant"] = bool(
            results["pair_order_max_abs_pred_diff"] < 1.0e-5
        )
    else:
        results["pair_order_max_abs_pred_diff"] = 0.0
        results["pair_order_invariant"] = True

    results["passed"] = bool(
        results["forward_without_pair_to_center"]
        and results["pair_encoder_calls_per_forward"] == 1
        and results["relation_encoder_calls_per_forward"] == 1
        and results["h_identical_under_relation_mutation"]
        and results["alpha_identical_under_relation_mutation"]
        and results["coord_identical_under_relation_mutation"]
        and results["prediction_changes_under_relation_mutation"]
        and results["pair_order_invariant"]
    )
    return results


def gradient_viability_checks(model: SDPKModel, batch: Any) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(batch).view(-1)
    target = batch.y.view(-1)
    loss = F.l1_loss(prediction, target)
    loss.backward()

    def grad_norm(parameters: Sequence[nn.Parameter]) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        return math.sqrt(total)

    payload: dict[str, Any] = {"loss": float(loss.detach())}
    payload["dictionary_atoms_grad_norm"] = grad_norm([model.dictionary.atoms])
    payload["wq_grad_norm"] = grad_norm(list(model.dictionary.query.parameters()))
    payload["u_grad_norm"] = grad_norm(list(model.dictionary.coord_proj.parameters()))
    payload["tau_grad_norm"] = grad_norm([model.dictionary.tau_logit])
    payload["wm_grad_norm"] = grad_norm(list(model.dict_gate.parameters()))
    payload["pair_encoder_grad_norm"] = grad_norm(list(model.pair_encoder.parameters()))
    payload["relation_encoder_grad_norm"] = grad_norm(list(model.relation_encoder.parameters()))
    payload["head_grad_norm"] = grad_norm(list(model.head.parameters()))
    model.eval()
    model.zero_grad(set_to_none=True)

    required_positive = (
        "dictionary_atoms_grad_norm",
        "wq_grad_norm",
        "u_grad_norm",
        "wm_grad_norm",
        "pair_encoder_grad_norm",
    )
    payload["all_finite"] = bool(
        all(
            value is None or math.isfinite(float(value))
            for key, value in payload.items()
            if key != "loss"
        )
    )
    payload["required_positive_gradients"] = list(required_positive)
    payload["required_positive_ok"] = bool(
        all(float(payload[key]) > 0.0 for key in required_positive)
    )
    payload["branch_receives_gradient"] = bool(
        payload["all_finite"] and payload["required_positive_ok"]
    )
    return payload


# ---------------------------------------------------------------------------
# data (inherit the encoded train/valid cache; official test never touched)
# ---------------------------------------------------------------------------


def prepare_encoded(force: bool = False) -> dict[str, Any]:
    return sdp.prepare_encoded(force=force)


def load_encoded(
    train_subset: int | None = None, valid_subset: int | None = None
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    return sdp.load_encoded(train_subset, valid_subset)


# ---------------------------------------------------------------------------
# training / evaluation (faithful mirror of the inherited protocol)
# ---------------------------------------------------------------------------


def _evaluate_mae(model: nn.Module, loader, device: torch.device):
    return sdp._evaluate_mae(model, loader, device)


def _topk_epochs(rows: Sequence[Mapping[str, Any]], key: str, k: int = 5) -> list[int]:
    return sdp._topk_epochs(rows, key, k)


def train_arm(
    *,
    seed: int = 0,
    device: str = "cpu",
    max_epochs: int = 240,
    train_subset: int | None = None,
    valid_subset: int | None = None,
    tag: str | None = None,
    save_state: bool = True,
    soup: bool = True,
) -> dict[str, Any]:
    """Full-horizon training (no early termination) of SDPK-v0."""
    device_obj = torch.device(str(device))
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    train_data, valid_data, audit = load_encoded(train_subset, valid_subset)
    if int(audit["typed_vocabulary_size_with_oov"]) != TYPED_VOCAB_SIZE:
        raise RuntimeError("typed vocabulary drift")
    if int(audit["parent_vocabulary_size_with_oov"]) != PARENT_VOCAB_SIZE:
        raise RuntimeError("parent vocabulary drift")

    protocol = dict(OPTIMIZED_PROTOCOL)
    protocol["max_epochs"] = int(max_epochs)
    protocol["early_termination"] = "none"

    model = build_sdpl(seed=seed).to(device_obj)
    total_params = _n_params(model)

    first_batch = _first_batch(valid_data, device_obj, limit=min(128, len(valid_data)))
    contract = static_contract_checks(model, first_batch)
    if not contract["passed"]:
        raise RuntimeError(f"strict-static integrity gates failed: {contract}")
    viability = gradient_viability_checks(model, first_batch)
    if not viability["branch_receives_gradient"]:
        raise RuntimeError(f"dictionary kernel receives no step-0 gradient: {viability}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    loader = zpp._make_loader(
        train_data,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = zpp._make_loader(
        valid_data,
        int(protocol["batch_size"]),
        False,
        int(seed) + int(protocol["eval_shuffle_seed_offset"]),
    )
    steps_per_epoch = int(math.ceil(len(train_data) / int(protocol["batch_size"])))
    clip = float(protocol["gradient_clip_norm"])

    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}

    started = time.perf_counter()
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        steps = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            steps += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae, _, _ = _evaluate_mae(model, eval_loader, device_obj)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_mae": float(train_mae),
            "valid_mae": float(valid_mae),
            "tau": float(model.dictionary.current_tau().detach()),
        }
        curve.append(row)
        epoch_states[epoch] = {
            key: value.detach().to("cpu", copy=True)
            for key, value in model.state_dict().items()
        }
        keep = set(_topk_epochs(curve, "valid_mae", 5))
        for cached_epoch in list(epoch_states):
            if cached_epoch not in keep:
                del epoch_states[cached_epoch]
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().to("cpu", copy=True)
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            print(
                f"[{tag or 'sdpl'} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch} "
                f"tau={row['tau']:.4f}",
                flush=True,
            )
    wall_clock = float(time.perf_counter() - started)

    for row in curve:
        row["checkpoint_selected"] = int(row["epoch"] == best_epoch)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device_obj)

    best_valid, valid_targets, best_predictions = _evaluate_mae(
        model, eval_loader, device_obj
    )

    soup_payload: dict[str, Any] = {"available": False}
    if soup and epoch_states:
        members = sorted(_topk_epochs(curve, "valid_mae", 5))
        keys = list(epoch_states[members[0]].keys())
        soup_state = {
            key: torch.stack([epoch_states[epoch][key].float() for epoch in members]).mean(0)
            for key in keys
        }
        soup_model = build_sdpl(seed=seed)
        soup_model.load_state_dict(soup_state)
        soup_model.to(device_obj)
        soup_mae, _, soup_predictions = _evaluate_mae(soup_model, eval_loader, device_obj)
        soup_payload = {
            "available": True,
            "members": [int(epoch) for epoch in members],
            "member_valid_mae": [float(curve[epoch - 1]["valid_mae"]) for epoch in members],
            "soup_valid_mae": float(soup_mae),
            "soup_predictions": soup_predictions.tolist(),
        }
        del soup_model

    diagnostics = dictionary_diagnostics(model, eval_loader, device_obj)
    intervention = dictionary_intervention_shift(
        model, eval_loader, device_obj, best_predictions
    )

    curve_path = CURVE_DIR / f"{tag or 'sdpl'}_seed{seed}_curve.csv"
    _write_csv(curve_path, curve)
    state_path = None
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"{tag or 'sdpl'}_seed{seed}_selection_state.pt"
        torch.save(model.state_dict(), state_path)

    peak_memory_mb = None
    if device_obj.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))

    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "seed": int(seed),
        "tag": str(tag or "sdpl"),
        "device": str(device_obj),
        "protocol": protocol,
        "max_epochs": int(max_epochs),
        "epochs_run": int(len(losses)),
        "early_terminated": bool(len(losses) < int(max_epochs)),
        "steps_per_epoch": int(steps_per_epoch),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(best_valid),
        "wall_clock_s": float(wall_clock),
        "peak_gpu_memory_mb": peak_memory_mb,
        "parameters": int(total_params),
        "tau_final": float(model.dictionary.current_tau().detach()),
        "curve_path": str(curve_path),
        "state_path": None if state_path is None else str(state_path),
        "strict_static_contract": contract,
        "gradient_viability": viability,
        "dictionary_diagnostics": diagnostics,
        "dictionary_intervention": intervention,
        "soup": soup_payload,
        "valid_predictions": best_predictions.tolist(),
        "valid_targets": valid_targets.tolist(),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(str(device_obj)),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RUNS_DIR / f"{tag or 'sdpl'}_seed{seed}.json", summary)
    print(
        f"[{tag or 'sdpl'} seed{seed}] best_valid={best_mae:.6f} epoch={best_epoch} "
        f"soup={soup_payload.get('soup_valid_mae')} params={total_params} "
        f"wall={wall_clock:.1f}s",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# cheap dictionary diagnostics (inference only; never a GO gate)
# ---------------------------------------------------------------------------


def _collect_local_states(model: SDPKModel, loader, device: torch.device):
    chunks: list[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        chunks.append(output.detach())

    handle = model.patch_encoder.register_forward_hook(_hook)
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                model(batch)
    finally:
        handle.remove()
    return torch.cat(chunks, dim=0)


def dictionary_diagnostics(
    model: SDPKModel, loader, device: torch.device
) -> dict[str, Any]:
    h = _collect_local_states(model, loader, device)
    alpha, tau, atoms = model.dictionary.alpha_from_h(h)
    eps = 1.0e-12
    entropy = -(alpha * (alpha + eps).log()).sum(dim=1)
    mean_mass = alpha.mean(dim=0)
    cos = atoms @ atoms.t()
    off = cos - torch.eye(cos.shape[0], device=cos.device, dtype=cos.dtype)
    top8 = torch.topk(alpha, k=min(8, alpha.shape[1]), dim=1).values.sum(dim=1)
    coord = model.dictionary.coord_proj(alpha)
    return {
        "mean_assignment_entropy": float(entropy.mean()),
        "effective_atom_count": float(torch.exp(entropy.mean())),
        "active_atom_count": int((mean_mass > 1.0e-3).sum()),
        "argmax_used_atoms": int(torch.unique(alpha.argmax(dim=1)).numel()),
        "top8_assignment_mass": float(top8.mean()),
        "max_average_assignment_mass": float(mean_mass.max()),
        "min_average_assignment_mass": float(mean_mass.min()),
        "dictionary_coherence_mean_abs": float(off.abs().mean()),
        "dictionary_coherence_max_abs": float(off.abs().max()),
        "tau_final": float(tau),
        "coord_norm_mean": float(coord.norm(dim=1).mean()),
        "coord_std": float(coord.std(dim=0).mean()),
        "n_patches": int(h.shape[0]),
    }


def dictionary_intervention_shift(
    model: SDPKModel,
    loader,
    device: torch.device,
    predictions_normal: np.ndarray,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"applicable": True}
    normal = predictions_normal.astype(np.float64)
    for mode in ("mean_coord", "neutral"):
        model.set_dict_intervention(mode)
        try:
            _mae, _targets, predictions = _evaluate_mae(model, loader, device)
        finally:
            model.set_dict_intervention(None)
        shift = np.abs(normal - predictions.astype(np.float64))
        payload[mode] = {
            "mean_abs_prediction_shift": float(shift.mean()),
            "max_abs_prediction_shift": float(shift.max()),
            "frac_shift_gt_1e-6": float((shift > 1.0e-6).mean()),
        }
    return payload


# ---------------------------------------------------------------------------
# baseline guard: shared S0 code path unchanged
# ---------------------------------------------------------------------------


def baseline_guard() -> dict[str, Any]:
    """Cheap guard that S0 source and shared tensors were not disturbed."""
    import hashlib

    s0 = sdp.build_s0(0)
    source_files = {
        "zinc_patch_path_pooling.py": TRACK_ROOT
        / "experiments/luyin16/zinc_patch_path_pooling.py",
        "zinc_static_dictionary_pair.py": TRACK_ROOT
        / "experiments/luyin16/zinc_static_dictionary_pair.py",
        "zinc_compact_v4_topology_hinge.yaml": TRACK_ROOT
        / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml",
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_files.items()
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "s0_params": _n_params(s0),
        "s0_center_update_is_none": bool(s0.center_update is None),
        "modified_by_this_round": False,
        "source_sha256": hashes,
        "sdpk_shared_tensors_bit_identical_to_s0": initialization_match()[
            "bit_identical_shared"
        ],
        "checkpoint_replay": {"attempted": False},
        "official_test_loaded": False,
    }

    s0_ckpt = sdp.STATE_DIR / "s0_seed0_selection_state.pt"
    if s0_ckpt.exists():
        train_data, valid_data, _audit = load_encoded(valid_subset=None)
        device = torch.device("cpu")
        model_s0 = sdp.build_s0(0)
        model_s0.load_state_dict(torch.load(s0_ckpt, map_location="cpu", weights_only=False))
        model_s0.to(device)
        eval_loader = zpp._make_loader(valid_data, 128, False, 91012)
        mae, _t, _p = _evaluate_mae(model_s0, eval_loader, device)
        payload["checkpoint_replay"] = {
            "attempted": True,
            "checkpoint": str(s0_ckpt),
            "replayed_valid_mae": float(mae),
            "recorded_valid_mae": float(S0_SEED0_BEST),
            "abs_diff": float(abs(mae - S0_SEED0_BEST)),
            "reproduced": bool(abs(mae - S0_SEED0_BEST) < 1.0e-6),
        }
    _write_json(RESULTS_DIR / "baseline_guard.json", payload)
    return payload


def integrity_stage() -> dict[str, Any]:
    _train_data, valid_data, _audit = load_encoded(valid_subset=64)
    batch = _first_batch(valid_data, torch.device("cpu"), limit=64)
    model = build_sdpl(0)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "contract": static_contract_checks(model, batch),
        "gradient_viability": gradient_viability_checks(model, batch),
        "official_test_loaded": False,
    }
    payload["passed"] = bool(
        payload["contract"]["passed"]
        and payload["gradient_viability"]["branch_receives_gradient"]
    )
    _write_json(RESULTS_DIR / "integrity_gates.json", payload)
    return payload


def smoke_stage(device: str = "cuda") -> dict[str, Any]:
    summary = train_arm(
        seed=0,
        device=device,
        max_epochs=2,
        train_subset=256,
        valid_subset=128,
        tag="smoke",
        save_state=False,
        soup=False,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "best_valid_mae": summary["best_valid_mae"],
        "parameters": summary["parameters"],
        "contract_passed": summary["strict_static_contract"]["passed"],
        "gradient_viability": summary["gradient_viability"],
        "dictionary_diagnostics": summary["dictionary_diagnostics"],
        "dictionary_intervention": summary["dictionary_intervention"],
        "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
        "device": summary["device"],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "smoke.json", payload)
    return payload


# ---------------------------------------------------------------------------
# analysis / gates
# ---------------------------------------------------------------------------


def classify_seed0(best: float, soup: float) -> dict[str, Any]:
    soup_delta = float(S0_SEED0_SOUP) - float(soup)
    best_delta = float(S0_SEED0_BEST) - float(best)
    go = bool(soup <= GATE_SEED0_SOUP and soup_delta >= GATE_SEED0_SOUP_DELTA
              and best_delta >= GATE_SEED0_BEST_DELTA)
    return {
        "reference_s0_best": float(S0_SEED0_BEST),
        "reference_s0_soup": float(S0_SEED0_SOUP),
        "sdpl_best": float(best),
        "sdpl_soup": float(soup),
        "best_improvement": best_delta,
        "soup_improvement": soup_delta,
        "gate_soup_abs": GATE_SEED0_SOUP,
        "gate_soup_delta": GATE_SEED0_SOUP_DELTA,
        "gate_best_delta": GATE_SEED0_BEST_DELTA,
        "go": go,
        "verdict": "GO_SEED1" if go else "SDPK_V0_NO_STRONG_PERFORMANCE_SIGNAL",
    }


def classify_seed1(best: float, soup: float) -> dict[str, Any]:
    soup_delta = float(S0_SEED1_SOUP) - float(soup)
    best_delta = float(S0_SEED1_BEST) - float(best)
    strong = bool(soup <= GATE_SEED1_SOUP or soup_delta >= GATE_SEED1_SOUP_DELTA)
    directional = bool(soup_delta > 0.0)
    if strong:
        verdict = "SEED1_STRONG_CONFIRMATION"
    elif directional:
        verdict = "DIRECTIONAL_ONLY"
    else:
        verdict = "SEED1_NO_SIGNAL"
    return {
        "reference_s0_best": float(S0_SEED1_BEST),
        "reference_s0_soup": float(S0_SEED1_SOUP),
        "sdpl_best": float(best),
        "sdpl_soup": float(soup),
        "best_improvement": best_delta,
        "soup_improvement": soup_delta,
        "gate_soup_abs": GATE_SEED1_SOUP,
        "gate_soup_delta": GATE_SEED1_SOUP_DELTA,
        "strong": strong,
        "verdict": verdict,
    }


def analyze_stage() -> dict[str, Any]:
    seed0_path = RUNS_DIR / "sdpl_seed0.json"
    if not seed0_path.exists():
        raise FileNotFoundError(f"missing seed0 run output: {seed0_path}")
    seed0 = _read_json(seed0_path)
    soup0 = seed0["soup"].get("soup_valid_mae")
    seed0_gate = classify_seed0(seed0["best_valid_mae"], soup0)

    seed1 = None
    seed1_gate = None
    seed1_path = RUNS_DIR / "sdpl_seed1.json"
    if seed1_path.exists():
        seed1 = _read_json(seed1_path)
        seed1_gate = classify_seed1(
            seed1["best_valid_mae"], seed1["soup"].get("soup_valid_mae")
        )

    runs_used = 1 + (1 if seed1 is not None else 0)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "primary_metric": "fixed Top-5 weight soup official-valid MAE",
        "secondary_metric": "best-checkpoint official-valid MAE",
        "seed0_gate": seed0_gate,
        "seed1_gate": seed1_gate,
        "seed0": {
            "best_valid_mae": seed0["best_valid_mae"],
            "best_epoch": seed0["best_epoch"],
            "epochs_run": seed0["epochs_run"],
            "soup_valid_mae": soup0,
            "soup_members": seed0["soup"].get("members"),
            "parameters": seed0["parameters"],
            "wall_clock_s": seed0["wall_clock_s"],
            "peak_gpu_memory_mb": seed0.get("peak_gpu_memory_mb"),
            "device": seed0["device"],
            "git_commit": seed0["git_commit"],
            "contract_passed": seed0["strict_static_contract"]["passed"],
            "dictionary_diagnostics": seed0["dictionary_diagnostics"],
            "dictionary_intervention": seed0["dictionary_intervention"],
        },
        "seed1": None
        if seed1 is None
        else {
            "best_valid_mae": seed1["best_valid_mae"],
            "best_epoch": seed1["best_epoch"],
            "epochs_run": seed1["epochs_run"],
            "soup_valid_mae": seed1["soup"].get("soup_valid_mae"),
            "soup_members": seed1["soup"].get("members"),
            "parameters": seed1["parameters"],
            "wall_clock_s": seed1["wall_clock_s"],
            "peak_gpu_memory_mb": seed1.get("peak_gpu_memory_mb"),
            "device": seed1["device"],
            "git_commit": seed1["git_commit"],
            "contract_passed": seed1["strict_static_contract"]["passed"],
            "dictionary_diagnostics": seed1["dictionary_diagnostics"],
            "dictionary_intervention": seed1["dictionary_intervention"],
        },
        "budget": {
            "max_full_runs": MAX_FULL_RUNS,
            "full_runs_used": runs_used,
            "seed1_purchased": bool(seed1 is not None),
            "matched_dense_control": False,
            "hpo": False,
        },
        "parameter_audit": parameter_audit(),
        "initialization_match": initialization_match(),
        "baseline_guard": _read_json(RESULTS_DIR / "baseline_guard.json")
        if (RESULTS_DIR / "baseline_guard.json").exists()
        else None,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "analysis.json", payload)
    _write_summary_markdown(payload)
    return payload


def _write_summary_markdown(payload: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# SDPK-v0 — static dictionary-conditioned pair kernel: results summary")
    lines.append("")
    lines.append(
        "Question: does promoting the learnable dictionary from a bypassable "
        "local residual adapter to the core coordinate of the occurrence-level "
        "static pair kernel open a new absolute strict-static MAE band?"
    )
    lines.append("")
    lines.append("Official ZINC **test was never loaded** in any stage.")
    lines.append("")
    lines.append("## Primary metric — fixed Top-5 weight soup (official-valid MAE)")
    lines.append("")
    lines.append("| run | best MAE | best epoch | epochs | soup MAE | params | wall s |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(
        f"| S0 seed0 (reference) | {S0_SEED0_BEST:.6f} | - | - | "
        f"{S0_SEED0_SOUP:.6f} | 66228 | (historical) |"
    )
    s0 = payload["seed0"]
    lines.append(
        f"| SDPK-v0 seed0 | {s0['best_valid_mae']:.6f} | {s0['best_epoch']} | "
        f"{s0['epochs_run']} | {s0['soup_valid_mae']:.6f} | {s0['parameters']} | "
        f"{s0['wall_clock_s']:.1f} |"
    )
    if payload["seed1"] is not None:
        s1 = payload["seed1"]
        lines.append(
            f"| S0 seed1 (reference) | {S0_SEED1_BEST:.6f} | - | - | "
            f"{S0_SEED1_SOUP:.6f} | 66228 | (historical) |"
        )
        lines.append(
            f"| SDPK-v0 seed1 | {s1['best_valid_mae']:.6f} | {s1['best_epoch']} | "
            f"{s1['epochs_run']} | {s1['soup_valid_mae']:.6f} | {s1['parameters']} | "
            f"{s1['wall_clock_s']:.1f} |"
        )
    lines.append("")
    gate0 = payload["seed0_gate"]
    lines.append("## Seed-0 performance gate")
    lines.append("")
    lines.append(f"* soup improvement over S0 seed0: {gate0['soup_improvement']:+.6f}")
    lines.append(f"* best improvement over S0 seed0: {gate0['best_improvement']:+.6f}")
    lines.append(f"* gate (soup <= {gate0['gate_soup_abs']} and delta >= {gate0['gate_soup_delta']} "
                 f"and best delta >= {gate0['gate_best_delta']}): "
                 f"**{'PASS' if gate0['go'] else 'FAIL'}**")
    lines.append(f"* verdict: **{gate0['verdict']}**")
    if payload.get("seed1_gate") is not None:
        gate1 = payload["seed1_gate"]
        lines.append("")
        lines.append("## Seed-1 conditional gate")
        lines.append("")
        lines.append(f"* soup improvement over S0 seed1: {gate1['soup_improvement']:+.6f}")
        lines.append(f"* best improvement over S0 seed1: {gate1['best_improvement']:+.6f}")
        lines.append(f"* verdict: **{gate1['verdict']}**")
    lines.append("")
    lines.append("## Budget")
    lines.append("")
    budget = payload["budget"]
    lines.append(f"* full training runs used: {budget['full_runs_used']} / {budget['max_full_runs']}")
    lines.append(f"* seed1 purchased: {budget['seed1_purchased']}")
    lines.append("* matched dense control: not purchased; HPO: none")
    lines.append("* official test accessed = false")
    lines.append("")
    (RESULTS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "param_audit",
            "init_match",
            "integrity",
            "encode",
            "baseline_guard",
            "smoke",
            "train",
            "analyze",
            "all",
        ],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--valid-subset", type=int, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)
    _configure_determinism()
    if args.stage == "param_audit":
        print(json.dumps(parameter_audit(), indent=2))
    elif args.stage == "init_match":
        print(json.dumps(initialization_match(), indent=2))
    elif args.stage == "integrity":
        print(json.dumps(integrity_stage(), indent=2))
    elif args.stage == "encode":
        print(json.dumps(prepare_encoded(), indent=2))
    elif args.stage == "baseline_guard":
        print(json.dumps(baseline_guard(), indent=2))
    elif args.stage == "smoke":
        print(json.dumps(smoke_stage(args.device), indent=2))
    elif args.stage == "train":
        summary = train_arm(
            seed=args.seed,
            device=args.device,
            max_epochs=args.epochs,
            train_subset=args.train_subset,
            valid_subset=args.valid_subset,
            tag=args.tag,
        )
        print(json.dumps({k: v for k, v in summary.items() if "predictions" not in k}, indent=2))
    elif args.stage == "analyze":
        print(json.dumps(analyze_stage(), indent=2))
    else:
        parameter_audit()
        initialization_match()
        prepare_encoded()
        integrity_stage()
        analyze_stage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
