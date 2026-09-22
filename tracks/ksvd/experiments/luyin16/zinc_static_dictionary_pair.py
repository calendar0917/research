"""Strict-static patch-pair baseline + residual dictionary vs generic dense adapter.

Round: **ZINC-strict-static-dictionary-pair-v0**.

Single question
---------------
On a modern, strongly-trained **strict-static** patch-pair baseline, is a
residual *learnable dictionary* more valuable than a parameter-matched
*generic dense* residual adapter?

Strict-static contract (the architecture contract of this round)
----------------------------------------------------------------
The forward pass may **not** perform message passing / pair-to-centre update /
relation-result write-back into any local hidden state::

    h_i' = Update(h_i, Aggregate_j(q_ij))        # FORBIDDEN

Allowed: static graph/patch preprocessing, a shared per-patch local encoder,
unordered patch-pair direct computation, relation/path descriptors, invariant
graph-level pooling, graph-level topology features, end-to-end backprop.

Concretely the baseline sets ``center_context=False`` so ``center_update is
None`` and ``_pool_pairs_to_centres`` is never called.  Every pair state
``q_ij`` is computed exactly once and is only ever pooled at graph level.

Arms (one seed-0 full run each; nothing else is purchased)
----------------------------------------------------------
* ``S0``      -- strict-static compact-v4-smallhead baseline.
* ``S-Dense`` -- S0 + ``Linear(48,H)->SiLU->Linear(H,48)`` residual adapter.
* ``S-Dict``  -- S0 + learnable residual dictionary (K=64, d=32) adapter.

The dictionary residual is::

    q_i = Wq(h0_i)                      # 48 -> 32
    q_i = l2_normalize(q_i)
    D_k = l2_normalize(dictionary_k)    # K x 32, forward normalization
    tau = 0.05 + 0.95 * sigmoid(tau_logit)
    alpha_i = softmax((q_i @ D^T) / tau)
    d_i = alpha_i @ D
    delta_i = Wv(d_i)                   # 32 -> 48
    h_i = h0_i + gamma * delta_i

with ``K=64``, ``d=32``, ``tau ~= 0.20`` at init, ``gamma = 0.1`` at init
(learnable).  The residual path always keeps ``h0_i`` and there is **no**
zero-init on the final projection (the repo has a history of zero-init
truncating upstream gradients and silently disabling a branch).

No dictionary regularization of any kind is applied; the training loss is the
inherited ``L1 / MAE``.  Prototype statistics are diagnostics only.  Official
ZINC test is never loaded.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair <stage>

Stages: ``param_audit integrity encode smoke s0 dense dict train analyze all``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency import (
    build_encoded,
    load_train_valid_records,
)
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
CANONICAL_V4_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
RECORD_CACHE_DIR = TRACK_ROOT / "results/post_v4_residual_audit/cache"

RESULTS_DIR = TRACK_ROOT / "results/zinc_static_dictionary_pair"
CACHE_DIR = RESULTS_DIR / "cache"
RUNS_DIR = RESULTS_DIR / "runs"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"

EXPERIMENT_NAME = "zinc_static_dictionary_pair"
PROTOCOL_VERSION = "zinc_strict_static_dict_pair_v0"

# Historical aliased rooted-topology tokenizer (performance-oriented; this is
# NOT an "exact typed token" and must be described as *historical aliased
# rooted-topology token* in every note).
TYPED_VOCAB_SIZE = 6785
PARENT_VOCAB_SIZE = 32

HEAD_HIDDEN = (13, 13)  # audited Sraw / Hsmall reader

# --- residual dictionary (fixed; no sweep) ---------------------------------
DICT_RANK = 32
DICT_ATOMS = 64
DICT_TAU_INIT = 0.20
RESIDUAL_GAMMA_INIT = 0.1

# --- inherited optimized training protocol (not re-tuned) ------------------
OPTIMIZED_PROTOCOL: dict[str, Any] = {
    "optimizer": "Adam",
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "batch_size": 128,
    "max_epochs": 240,
    "patience": 40,
    "scheduler": "none",
    "gradient_clip_norm": 5.0,
    "loss": "L1 / mean absolute error",
    "checkpoint_selection": "best official-valid MAE",
    "single_stage": True,
    "train_shuffle_seed_offset": 91011,
    "eval_shuffle_seed_offset": 91012,
}
TORCH_THREADS = 4

# --- pre-registered interpretation gates (seed-0 exploratory) --------------
GATE_LOCAL_GAIN = 0.004      # M0 - MK >= 0.004 -> meaningful local gain
GATE_DICT_SPECIFIC = 0.002   # MD - MK >= 0.002 -> dictionary-specific edge


# ---------------------------------------------------------------------------
# io / determinism helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row.get(field, "")) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _n_params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - untracked checkout
        return "unknown"


def _environment_fingerprint(device: str = "cpu") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
    }
    if torch.cuda.is_available():
        payload["cuda_available"] = True
        payload["torch_cuda"] = torch.version.cuda
        payload["gpu_count"] = int(torch.cuda.device_count())
        payload["gpu_names"] = [
            torch.cuda.get_device_properties(i).name
            for i in range(torch.cuda.device_count())
        ]
    else:
        payload["cuda_available"] = False
    return payload


def _configure_determinism() -> None:
    torch.set_num_threads(int(TORCH_THREADS))
    # The inherited ZINC protocol is seeded but not strictly deterministic on
    # CUDA (index_add_ scatter).  All three arms share this execution regime.
    torch.use_deterministic_algorithms(False)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


# ---------------------------------------------------------------------------
# residual branches
# ---------------------------------------------------------------------------


def _tau_from_logit(tau_logit: torch.Tensor) -> torch.Tensor:
    return 0.05 + 0.95 * torch.sigmoid(tau_logit)


def _tau_logit_for(tau: float) -> float:
    fraction = (float(tau) - 0.05) / 0.95
    if not 0.0 < fraction < 1.0:
        raise ValueError("tau init must be strictly inside (0.05, 1.0)")
    return float(math.log(fraction / (1.0 - fraction)))


class ResidualDictionaryBranch(nn.Module):
    """Learnable residual dictionary: ``h0 -> delta`` (32-D code over K atoms).

    The dictionary is forward-normalized (unit-norm atoms); ``Wv`` is a normal
    small random projection.  There is deliberately **no** zero-init on the
    final projection.
    """

    def __init__(
        self,
        hidden: int,
        rank: int = DICT_RANK,
        atoms: int = DICT_ATOMS,
        tau_init: float = DICT_TAU_INIT,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.atoms_count = int(atoms)
        self.query = nn.Linear(self.hidden, self.rank)
        self.atoms = nn.Parameter(torch.empty(self.atoms_count, self.rank))
        self.value = nn.Linear(self.rank, self.hidden)
        self.tau_logit = nn.Parameter(torch.tensor(_tau_logit_for(tau_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.query.reset_parameters()
        self.value.reset_parameters()
        nn.init.normal_(self.atoms, std=1.0 / math.sqrt(self.rank))

    def current_tau(self) -> torch.Tensor:
        return _tau_from_logit(self.tau_logit)

    def normalized_atoms(self) -> torch.Tensor:
        return F.normalize(self.atoms, dim=-1, eps=1.0e-8)

    def alpha_from_h0(self, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = F.normalize(self.query(h0), dim=-1, eps=1.0e-8)
        atoms = self.normalized_atoms()
        tau = self.current_tau()
        logits = (query @ atoms.t()) / tau
        alpha = F.softmax(logits, dim=-1)
        return alpha, tau, atoms

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        alpha, _tau, atoms = self.alpha_from_h0(h0)
        code = alpha @ atoms
        return self.value(code)


class DenseResidualBranch(nn.Module):
    """Parameter-matched generic dense residual: ``h0 -> SiLU MLP -> delta``."""

    def __init__(self, hidden: int, dense_hidden: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.dense_hidden = int(dense_hidden)
        self.net = nn.Sequential(
            nn.Linear(self.hidden, self.dense_hidden),
            nn.SiLU(),
            nn.Linear(self.dense_hidden, self.hidden),
        )

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        return self.net(h0)


class ResidualPatchEncoder(nn.Module):
    """Shared local patch encoder, optionally followed by a residual adapter."""

    def __init__(
        self,
        base: nn.Module,
        branch: nn.Module | None,
        gamma_init: float = RESIDUAL_GAMMA_INIT,
    ) -> None:
        super().__init__()
        self.base = base
        self.branch = branch
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.residual_enabled = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        h0 = self.base(value)
        if self.branch is None or not self.residual_enabled:
            return h0
        delta = self.branch(h0)
        return h0 + self.gamma * delta


def _dict_branch_params(hidden: int = 48) -> int:
    return int(
        hidden * DICT_RANK
        + DICT_RANK
        + DICT_ATOMS * DICT_RANK
        + DICT_RANK * hidden
        + hidden
        + 1  # tau_logit
    )


def _dense_branch_params(hidden: int, dense_hidden: int) -> int:
    return int(
        hidden * dense_hidden
        + dense_hidden
        + dense_hidden * hidden
        + hidden
    )


def matched_dense_hidden(hidden: int = 48) -> tuple[int, int, int, float]:
    """Smallest-relative-error dense width matching the dictionary branch."""
    target = _dict_branch_params(hidden)
    best: tuple[int, int, int, float] | None = None
    for candidate in range(1, 512):
        params = _dense_branch_params(hidden, candidate)
        error = abs(params - target)
        relative = error / float(target)
        if best is None or error < best[1]:
            best = (candidate, error, params, relative)
    assert best is not None
    return best


DENSE_HIDDEN, _DENSE_ERR, _DENSE_PARAMS, _DENSE_REL = matched_dense_hidden(48)


# ---------------------------------------------------------------------------
# strict-static model
# ---------------------------------------------------------------------------


def base_model_kwargs() -> dict[str, Any]:
    """Inherit the mature compact-v4-smallhead architecture, minus the centre update."""
    config = yaml.safe_load(CANONICAL_V4_CONFIG.read_text(encoding="utf-8"))
    model_config = config["model"]
    return dict(
        patch_hidden=int(model_config["patch_hidden"]),
        pair_hidden=int(model_config["pair_hidden"]),
        token_width=int(model_config["token_width"]),
        dropout=float(model_config["dropout"]),
        embedding_mode=str(model_config["embedding_mode"]),
        embedding_rank=int(model_config["embedding_rank"]),
        hybrid_full_typed_tokens=(
            None
            if model_config.get("hybrid_full_typed_tokens") is None
            else int(model_config["hybrid_full_typed_tokens"])
        ),
        hybrid_full_parent_tokens=(
            None
            if model_config.get("hybrid_full_parent_tokens") is None
            else int(model_config["hybrid_full_parent_tokens"])
        ),
        # THE strict-static change: no pair -> centre/patch feedback at all.
        center_context=False,
        graph_head_hidden_0=None,
        graph_head_hidden_1=None,
        topology_mode=str(model_config["topology_mode"]),
        topology_input_width=int(ztopo.raw_width(str(model_config["topology_mode"]))),
        topology_hidden_dim=int(model_config["topology_hidden_dim"]),
        topology_out_dim=int(model_config["topology_out_dim"]),
    )


class StrictStaticPairModel(zpp.PatchPathModel):
    """compact-v4 patch-pair backbone with ``center_context=False``.

    The single graph head is the fixed small raw reader ``R -> 13 -> 13 -> 1``
    over the *true runtime* ``unified_graph_width`` (never hardcoded).

    ``residual_mode`` selects the residual adapter inserted directly on the
    shared local encoder output:

    * ``none``  -- pure strict-static S0, no adapter.
    * ``dense`` -- generic dense SiLU residual MLP.
    * ``dict``  -- learnable residual dictionary.
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        residual_mode: str = "none",
        dense_hidden: int = DENSE_HIDDEN,
        dict_rank: int = DICT_RANK,
        dict_atoms: int = DICT_ATOMS,
        dict_tau_init: float = DICT_TAU_INIT,
        gamma_init: float = RESIDUAL_GAMMA_INIT,
        head_hidden: Sequence[int] = HEAD_HIDDEN,
        **kwargs: Any,
    ) -> None:
        kwargs = dict(kwargs)
        kwargs["center_context"] = False
        super().__init__(
            int(typed_vocabulary_size), int(parent_vocabulary_size), **kwargs
        )
        if self.center_update is not None:  # pragma: no cover - contract guard
            raise RuntimeError("strict-static model must have center_update=None")
        self.residual_mode = str(residual_mode)
        base = self.patch_encoder
        branch: nn.Module | None
        if self.residual_mode == "none":
            branch = None
        elif self.residual_mode == "dense":
            branch = DenseResidualBranch(int(self.patch_hidden), int(dense_hidden))
        elif self.residual_mode == "dict":
            branch = ResidualDictionaryBranch(
                int(self.patch_hidden), int(dict_rank), int(dict_atoms), float(dict_tau_init)
            )
        else:
            raise ValueError(f"unknown residual_mode={self.residual_mode!r}")
        self.patch_encoder = ResidualPatchEncoder(base, branch, gamma_init)
        self.head = GenericReader(int(self.unified_graph_width), tuple(int(h) for h in head_hidden))

    # -- diagnostics helpers -------------------------------------------------
    def set_residual_enabled(self, flag: bool) -> "StrictStaticPairModel":
        self.patch_encoder.residual_enabled = bool(flag)
        return self

    @property
    def residual_branch(self) -> nn.Module | None:
        return self.patch_encoder.branch


def _build_strict_static(
    seed: int,
    *,
    mode: str = "none",
    typed_vocabulary_size: int = TYPED_VOCAB_SIZE,
    parent_vocabulary_size: int = PARENT_VOCAB_SIZE,
    head_seed: int = 0,
    branch_seed: int = 1,
    **branch_overrides: Any,
) -> StrictStaticPairModel:
    """Build one arm with shared tensors exactly matched to the S0 reference."""
    _seed_everything(int(seed))
    reference = StrictStaticPairModel(
        int(typed_vocabulary_size),
        int(parent_vocabulary_size),
        residual_mode="none",
        **base_model_kwargs(),
    )
    reference_state = {
        key: value.detach().clone() for key, value in reference.state_dict().items()
    }

    _seed_everything(int(seed))
    model = StrictStaticPairModel(
        int(typed_vocabulary_size),
        int(parent_vocabulary_size),
        residual_mode=str(mode),
        **branch_overrides,
        **base_model_kwargs(),
    )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in reference_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)

    # Independent, deterministic reader + branch initialization.
    torch.manual_seed(int(head_seed))
    model.head = GenericReader(int(model.unified_graph_width), HEAD_HIDDEN)
    if model.residual_branch is not None:
        torch.manual_seed(int(branch_seed))
        model.residual_branch.reset_parameters()
    return model


def build_s0(seed: int = 0, **kwargs: Any) -> StrictStaticPairModel:
    return _build_strict_static(seed, mode="none", **kwargs)


def build_dense(seed: int = 0, **kwargs: Any) -> StrictStaticPairModel:
    return _build_strict_static(seed, mode="dense", **kwargs)


def build_dict(seed: int = 0, **kwargs: Any) -> StrictStaticPairModel:
    return _build_strict_static(seed, mode="dict", **kwargs)


ARMS: dict[str, Any] = {
    "s0": build_s0,
    "dense": build_dense,
    "dict": build_dict,
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def prepare_encoded(force: bool = False) -> dict[str, Any]:
    """Extract + encode the official train/valid splits once and cache them."""
    train_path = CACHE_DIR / "encoded_train.pt"
    valid_path = CACHE_DIR / "encoded_valid.pt"
    audit_path = CACHE_DIR / "encoded_audit.json"
    if not force and train_path.exists() and valid_path.exists() and audit_path.exists():
        return _read_json(audit_path)
    train_records, valid_records = load_train_valid_records()
    config = yaml.safe_load(CANONICAL_V4_CONFIG.read_text(encoding="utf-8"))
    config["test_policy"] = "no_test"
    config["model"]["device"] = "cpu"
    train_data, valid_data, audit = build_encoded(train_records, valid_records, config)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(list(train_data), train_path)
    torch.save(list(valid_data), valid_path)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_train": int(len(train_data)),
        "n_valid": int(len(valid_data)),
        "typed_vocabulary_size_with_oov": int(audit["typed_vocabulary_size_with_oov"]),
        "parent_vocabulary_size_with_oov": int(audit["parent_vocabulary_size_with_oov"]),
        "patch_radius": int(audit["patch_radius"]),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(audit_path, payload)
    return payload


def load_encoded(
    train_subset: int | None = None, valid_subset: int | None = None
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train_path = CACHE_DIR / "encoded_train.pt"
    valid_path = CACHE_DIR / "encoded_valid.pt"
    audit_path = CACHE_DIR / "encoded_audit.json"
    if not (train_path.exists() and valid_path.exists() and audit_path.exists()):
        raise FileNotFoundError(
            "encoded cache missing; run the `encode` stage first"
        )
    train_data = list(torch.load(train_path, weights_only=False))
    valid_data = list(torch.load(valid_path, weights_only=False))
    audit = _read_json(audit_path)
    if train_subset is not None:
        train_data = train_data[: int(train_subset)]
    if valid_subset is not None:
        valid_data = valid_data[: int(valid_subset)]
    return train_data, valid_data, audit


# ---------------------------------------------------------------------------
# strict-static integrity + gradient viability
# ---------------------------------------------------------------------------


def _contract_violation(*_args: Any, **_kwargs: Any) -> torch.Tensor:
    raise RuntimeError(
        "strict-static contract violated: _pool_pairs_to_centres was called"
    )


def _first_batch(data: Sequence[Any], device: torch.device, limit: int = 128):
    loader = zpp._make_loader(list(data)[: int(limit)], int(limit), False, 0)
    return next(iter(loader)).to(device)


def static_contract_checks(
    model: StrictStaticPairModel, batch: Any
) -> dict[str, Any]:
    """Programmatic hard checks of the strict-static contract on a real batch."""
    model.eval()
    results: dict[str, Any] = {}

    results["center_context_false"] = bool(model.center_context is False)
    results["center_update_is_none"] = bool(model.center_update is None)
    if not results["center_context_false"] or not results["center_update_is_none"]:
        raise RuntimeError("strict-static architecture contract failed (center update present)")

    # (A) monkeypatch _pool_pairs_to_centres to raise; forward must still succeed.
    original = zpp.PatchPathModel._pool_pairs_to_centres
    zpp.PatchPathModel._pool_pairs_to_centres = _contract_violation
    try:
        with torch.no_grad():
            _ = model(batch)
        results["forward_without_pair_to_center"] = True
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original

    # (C) every pair is evaluated exactly once and never refreshed.
    counters = {"pair_encoder": 0, "relation_encoder": 0}
    handles = [
        model.pair_encoder.register_forward_hook(
            lambda *_a, _c=counters: _c.__setitem__("pair_encoder", _c["pair_encoder"] + 1)
        ),
        model.relation_encoder.register_forward_hook(
            lambda *_a, _c=counters: _c.__setitem__(
                "relation_encoder", _c["relation_encoder"] + 1
            )
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
    if counters["pair_encoder"] != 1:
        raise RuntimeError("pair encoder was evaluated more than once per forward")

    # (B) no pair -> local feedback: h_i must be bit-identical under pair mutation.
    captured: dict[str, torch.Tensor] = {}

    def _capture_h0(_module, _inputs, output):
        captured["h0"] = output.detach().clone()

    def _capture_h(_module, _inputs, output):
        captured["h"] = output.detach().clone()

    handle_h0 = model.patch_encoder.base.register_forward_hook(_capture_h0)
    handle_h = model.patch_encoder.register_forward_hook(_capture_h)
    try:
        with torch.no_grad():
            pred_before = model(batch)
        h_before = captured["h"].clone()
        h0_before = captured["h0"].clone()

        mutated = batch.clone()
        if int(mutated.pair_relation.shape[0]) > 0:
            generator = torch.Generator(device="cpu").manual_seed(1234)
            mutated.pair_relation = torch.randn(
                mutated.pair_relation.shape, generator=generator
            ).to(mutated.pair_relation.device)
        with torch.no_grad():
            pred_after = model(mutated)
        h_after = captured["h"].clone()
        h0_after = captured["h0"].clone()
    finally:
        handle_h0.remove()
        handle_h.remove()

    results["h0_identical_under_relation_mutation"] = bool(torch.equal(h0_before, h0_after))
    results["h_identical_under_relation_mutation"] = bool(torch.equal(h_before, h_after))
    results["max_abs_h_diff"] = float((h_before - h_after).abs().max().item())
    results["max_abs_h0_diff"] = float((h0_before - h0_after).abs().max().item())
    results["prediction_changes_under_relation_mutation"] = bool(
        float((pred_before - pred_after).abs().max().item()) > 1.0e-8
    )
    if not results["h_identical_under_relation_mutation"]:
        raise RuntimeError("strict-static contract violated: pair state feeds back into h_i")

    # Pair ordering permutation invariance (float tolerance: scatter sums).
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
        and results["h_identical_under_relation_mutation"]
        and results["pair_order_invariant"]
    )
    return results


def gradient_viability_checks(
    model: StrictStaticPairModel, batch: Any
) -> dict[str, Any]:
    """A real mini-batch backward: every branch parameter must receive task gradient."""
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

    payload: dict[str, Any] = {
        "loss": float(loss.detach()),
        "gamma_grad": (
            None if model.patch_encoder.gamma.grad is None else float(model.patch_encoder.gamma.grad)
        ),
    }
    branch = model.residual_branch
    if isinstance(branch, ResidualDictionaryBranch):
        payload.update(
            dictionary_grad_norm=grad_norm([branch.atoms]),
            wq_grad_norm=grad_norm(list(branch.query.parameters())),
            wv_grad_norm=grad_norm(list(branch.value.parameters())),
            tau_grad_norm=grad_norm([branch.tau_logit]),
        )
    elif isinstance(branch, DenseResidualBranch):
        payload.update(
            dense_layer0_grad_norm=grad_norm(list(branch.net[0].parameters())),
            dense_layer1_grad_norm=grad_norm(list(branch.net[2].parameters())),
        )
    model.eval()
    model.zero_grad(set_to_none=True)
    finite = all(
        value is None or math.isfinite(float(value))
        for key, value in payload.items()
        if key != "loss"
    )
    positive = True
    # Branch parameters must receive a strictly positive task gradient at step 0.
    # ``gamma`` may legitimately have a negative gradient, so it is only required
    # to be finite.
    for key in (
        "dictionary_grad_norm",
        "wq_grad_norm",
        "wv_grad_norm",
        "dense_layer0_grad_norm",
        "dense_layer1_grad_norm",
    ):
        if key in payload and payload[key] is not None and float(payload[key]) <= 0.0:
            positive = False
    gamma_ok = payload.get("gamma_grad") is not None and math.isfinite(
        float(payload["gamma_grad"])
    )
    payload["gamma_grad_finite"] = bool(gamma_ok)
    payload["all_finite"] = bool(finite)
    payload["branch_receives_gradient"] = bool(positive and gamma_ok)
    return payload


# ---------------------------------------------------------------------------
# training / evaluation (faithful mirror of the inherited protocol)
# ---------------------------------------------------------------------------


def _evaluate_mae(model: nn.Module, loader, device: torch.device):
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    if not targets:
        return float("nan"), np.zeros(0), np.zeros(0)
    t = np.concatenate(targets).astype(np.float64)
    p = np.concatenate(preds).astype(np.float64)
    return float(np.mean(np.abs(t - p))), t, p


def _topk_epochs(rows: Sequence[Mapping[str, Any]], key: str, k: int = 5) -> list[int]:
    """1-based epoch ids with the smallest ``key`` (fixed Top-k soup members)."""
    order = sorted(range(len(rows)), key=lambda i: float(rows[i][key]))
    return [int(i) + 1 for i in order[: int(k)]]


def train_arm(
    arm: str,
    *,
    seed: int = 0,
    device: str = "cpu",
    max_epochs: int | None = None,
    patience: int | None = None,
    train_subset: int | None = None,
    valid_subset: int | None = None,
    tag: str | None = None,
    save_state: bool = True,
    soup: bool = True,
) -> dict[str, Any]:
    device_obj = torch.device(str(device))
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    train_data, valid_data, audit = load_encoded(train_subset, valid_subset)
    if int(audit["typed_vocabulary_size_with_oov"]) != TYPED_VOCAB_SIZE:
        raise RuntimeError(
            f"typed vocabulary drift: {audit['typed_vocabulary_size_with_oov']} "
            f"!= {TYPED_VOCAB_SIZE}"
        )
    if int(audit["parent_vocabulary_size_with_oov"]) != PARENT_VOCAB_SIZE:
        raise RuntimeError("parent vocabulary drift")

    protocol = dict(OPTIMIZED_PROTOCOL)
    if max_epochs is not None:
        protocol["max_epochs"] = int(max_epochs)
    if patience is not None:
        protocol["patience"] = int(patience)

    model = ARMS[str(arm)](seed=seed).to(device_obj)
    total_params = _n_params(model)
    branch_params = _n_params(model.residual_branch)

    # Contract + gradient viability on a real mini-batch (before any step).
    first_batch = _first_batch(valid_data, device_obj, limit=min(128, len(valid_data)))
    contract = static_contract_checks(model, first_batch)
    if not contract["passed"]:
        raise RuntimeError(f"strict-static integrity gates failed for arm {arm}: {contract}")
    if not model.patch_encoder.residual_enabled:
        raise RuntimeError("residual unexpectedly disabled")

    viability = None
    if arm in ("dense", "dict"):
        viability = gradient_viability_checks(model, first_batch)
        if not viability["branch_receives_gradient"]:
            # Retry with a fresh branch seed before aborting: initialisation must
            # not leave the branch without a step-0 task gradient.
            torch.manual_seed(int(seed) + 17)
            model.residual_branch.reset_parameters()
            viability = gradient_viability_checks(model, first_batch)
            if not viability["branch_receives_gradient"]:
                raise RuntimeError(f"branch receives no step-0 gradient for arm {arm}")

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
    max_epochs = int(protocol["max_epochs"])
    patience = int(protocol["patience"])
    clip = float(protocol["gradient_clip_norm"])

    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    branch_params_list = (
        [] if model.residual_branch is None else list(model.residual_branch.parameters())
    )
    branch_ids = {id(p) for p in branch_params_list}

    def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        return math.sqrt(total)

    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        head_grad = 0.0
        branch_grad = 0.0
        backbone_grad = 0.0
        steps = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            head_grad += _grad_norm(head_params)
            branch_grad += _grad_norm(branch_params_list)
            backbone_grad += _grad_norm(
                [p for p in model.parameters() if id(p) not in head_ids and id(p) not in branch_ids]
            )
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
            "head_grad_norm": float(head_grad / max(steps, 1)),
            "branch_grad_norm": float(branch_grad / max(steps, 1)),
            "backbone_grad_norm": float(backbone_grad / max(steps, 1)),
            "gamma": float(model.patch_encoder.gamma.detach()),
        }
        if isinstance(model.residual_branch, ResidualDictionaryBranch):
            row["tau"] = float(model.residual_branch.current_tau().detach())
        curve.append(row)
        # Keep a bounded cache of weight snapshots so the fixed Top-5 soup can be
        # formed without a soup framework.
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
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == max_epochs:
            print(
                f"[{tag or arm} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch} "
                f"branch_g={row['branch_grad_norm']:.3e} gamma={row['gamma']:.4f}",
                flush=True,
            )
        if stale >= patience:
            print(f"[{tag or arm} seed{seed}] early_stop epoch={epoch}", flush=True)
            break
    wall_clock = float(time.perf_counter() - started)

    for row in curve:
        row["checkpoint_selected"] = int(row["epoch"] == best_epoch)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device_obj)

    best_valid, valid_targets, best_predictions = _evaluate_mae(model, eval_loader, device_obj)

    soup_payload: dict[str, Any] = {"available": False}
    if soup and epoch_states:
        members = sorted(_topk_epochs(curve, "valid_mae", 5))
        keys = list(epoch_states[members[0]].keys())
        soup_state = {
            key: torch.stack([epoch_states[epoch][key].float() for epoch in members]).mean(0)
            for key in keys
        }
        soup_model = ARMS[str(arm)](seed=seed)
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

    diagnostics: dict[str, Any] = {}
    if arm == "dict":
        diagnostics = dictionary_diagnostics(model, eval_loader, device_obj)
    residual_ablation = residual_ablation_shift(model, eval_loader, device_obj, best_predictions)

    curve_path = CURVE_DIR / f"{tag or arm}_seed{seed}_curve.csv"
    _write_csv(curve_path, curve)
    state_path = None
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"{tag or arm}_seed{seed}_selection_state.pt"
        torch.save(model.state_dict(), state_path)

    peak_memory_mb = None
    if device_obj.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))

    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "arm": str(arm),
        "seed": int(seed),
        "tag": str(tag or arm),
        "device": str(device_obj),
        "protocol": protocol,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "epochs_run": int(len(losses)),
        "early_stopped": bool(len(losses) < max_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(best_valid),
        "train_mae_at_best": float(losses[best_epoch - 1]),
        "wall_clock_s": float(wall_clock),
        "peak_gpu_memory_mb": peak_memory_mb,
        "parameters": int(total_params),
        "branch_parameters": int(branch_params),
        "branch_parameter_delta_vs_s0": int(branch_params),
        "gamma_final": float(model.patch_encoder.gamma.detach()),
        "tau_final": (
            float(model.residual_branch.current_tau().detach())
            if isinstance(model.residual_branch, ResidualDictionaryBranch)
            else None
        ),
        "curve_path": str(curve_path),
        "state_path": None if state_path is None else str(state_path),
        "strict_static_contract": contract,
        "gradient_viability": viability,
        "dictionary_diagnostics": diagnostics,
        "residual_ablation": residual_ablation,
        "soup": soup_payload,
        "valid_predictions": best_predictions.tolist(),
        "valid_targets": valid_targets.tolist(),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(str(device_obj)),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RUNS_DIR / f"{tag or arm}_seed{seed}.json", summary)
    print(
        f"[{tag or arm} seed{seed}] best_valid={best_mae:.6f} epoch={best_epoch} "
        f"params={total_params} branch_params={branch_params} wall={wall_clock:.1f}s",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def _collect_local_states(model: StrictStaticPairModel, loader, device: torch.device):
    h0_chunks: list[torch.Tensor] = []
    h_chunks: list[torch.Tensor] = []

    def _hook_h0(_module, _inputs, output):
        h0_chunks.append(output.detach())

    def _hook_h(_module, _inputs, output):
        h_chunks.append(output.detach())

    handles = [
        model.patch_encoder.base.register_forward_hook(_hook_h0),
        model.patch_encoder.register_forward_hook(_hook_h),
    ]
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(h0_chunks, dim=0), torch.cat(h_chunks, dim=0)


def dictionary_diagnostics(
    model: StrictStaticPairModel, loader, device: torch.device
) -> dict[str, Any]:
    branch = model.residual_branch
    if not isinstance(branch, ResidualDictionaryBranch):
        return {}
    h0, h = _collect_local_states(model, loader, device)
    alpha, tau, atoms = branch.alpha_from_h0(h0)
    eps = 1.0e-12
    entropy = -(alpha * (alpha + eps).log()).sum(dim=1)
    mean_mass = alpha.mean(dim=0)
    cos = atoms @ atoms.t()
    off = cos - torch.eye(cos.shape[0], device=cos.device, dtype=cos.dtype)
    top8 = torch.topk(alpha, k=min(8, alpha.shape[1]), dim=1).values.sum(dim=1)
    residual = model.patch_encoder.gamma.detach() * (
        h - h0
    )
    return {
        "mean_assignment_entropy": float(entropy.mean()),
        "effective_prototype_count": float(torch.exp(entropy.mean())),
        "active_prototype_count": int((mean_mass > 1.0e-3).sum()),
        "used_prototype_count_argmax": int(
            torch.unique(alpha.argmax(dim=1)).numel()
        ),
        "top8_assignment_mass": float(top8.mean()),
        "max_average_assignment_mass": float(mean_mass.max()),
        "min_average_assignment_mass": float(mean_mass.min()),
        "dictionary_coherence_mean_abs": float(off.abs().mean()),
        "dictionary_coherence_max_abs": float(off.abs().max()),
        "mean_residual_norm": float(residual.norm(dim=1).mean()),
        "mean_h0_norm": float(h0.norm(dim=1).mean()),
        "mean_h_norm": float(h.norm(dim=1).mean()),
        "tau_final": float(tau),
        "n_patches": int(h0.shape[0]),
    }


def residual_ablation_shift(
    model: StrictStaticPairModel,
    loader,
    device: torch.device,
    predictions_on: np.ndarray,
) -> dict[str, Any]:
    if model.residual_branch is None:
        return {"applicable": False}
    model.set_residual_enabled(False)
    try:
        _mae, _targets, predictions_off = _evaluate_mae(model, loader, device)
    finally:
        model.set_residual_enabled(True)
    shift = np.abs(predictions_on.astype(np.float64) - predictions_off.astype(np.float64))
    return {
        "applicable": True,
        "mean_abs_prediction_shift": float(shift.mean()),
        "max_abs_prediction_shift": float(shift.max()),
        "frac_shift_gt_1e-6": float((shift > 1.0e-6).mean()),
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def parameter_audit() -> dict[str, Any]:
    s0 = build_s0(0)
    dense = build_dense(0)
    dict_model = build_dict(0)
    r_dim = int(s0.unified_graph_width)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "unified_graph_width": r_dim,
        "unified_graph_width_is_302": bool(r_dim == 302),
        "head_hidden": list(HEAD_HIDDEN),
        "center_context": bool(s0.center_context),
        "center_update_is_none": bool(s0.center_update is None),
        "patch_hidden": int(s0.patch_hidden),
        "pair_hidden": int(s0.pair_hidden),
        "token_width": int(s0.token_width),
        "params": {
            "s0_total": _n_params(s0),
            "s0_head": _n_params(s0.head),
            "dense_total": _n_params(dense),
            "dict_total": _n_params(dict_model),
            "dict_branch": _n_params(dict_model.residual_branch),
            "dense_branch": _n_params(dense.residual_branch),
        },
        "dense_hidden_matched": int(DENSE_HIDDEN),
        "dict_spec": {"rank": DICT_RANK, "atoms": DICT_ATOMS, "tau_init": DICT_TAU_INIT},
        "parameter_match": {
            "dict_branch_params": int(_n_params(dict_model.residual_branch)),
            "dense_branch_params": int(_n_params(dense.residual_branch)),
            "abs_diff": int(_n_params(dict_model.residual_branch))
            - int(_n_params(dense.residual_branch)),
            "relative_error": abs(
                int(_n_params(dense.residual_branch))
                - int(_n_params(dict_model.residual_branch))
            )
            / float(int(_n_params(dict_model.residual_branch))),
            "within_one_percent": bool(
                abs(
                    int(_n_params(dense.residual_branch))
                    - int(_n_params(dict_model.residual_branch))
                )
                / float(int(_n_params(dict_model.residual_branch)))
                < 0.01
            ),
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


def initialization_match() -> dict[str, Any]:
    s0 = build_s0(0)
    dense = build_dense(0)
    dict_model = build_dict(0)
    s0_state = s0.state_dict()
    dense_state = dense.state_dict()
    dict_state = dict_model.state_dict()
    branch_keys = {
        key
        for key in dict_state
        if key.startswith("patch_encoder.branch.") or key == "patch_encoder.gamma"
    }
    shared_keys = sorted(
        key
        for key in s0_state
        if key in dense_state
        and key in dict_state
        and s0_state[key].shape == dense_state[key].shape == dict_state[key].shape
        and key not in branch_keys
    )
    max_s0_dense = max(
        float((s0_state[key] - dense_state[key]).abs().max().item()) for key in shared_keys
    )
    max_s0_dict = max(
        float((s0_state[key] - dict_state[key]).abs().max().item()) for key in shared_keys
    )
    max_dense_dict = max(
        float((dense_state[key] - dict_state[key]).abs().max().item()) for key in shared_keys
    )
    head_keys = [key for key in shared_keys if key.startswith("head.")]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_shared_tensors": int(len(shared_keys)),
        "n_head_tensors": int(len(head_keys)),
        "shared_keys": shared_keys,
        "max_abs_diff_s0_vs_dense": max_s0_dense,
        "max_abs_diff_s0_vs_dict": max_s0_dict,
        "max_abs_diff_dense_vs_dict": max_dense_dict,
        "bit_identical": bool(max_s0_dense == 0.0 and max_s0_dict == 0.0 and max_dense_dict == 0.0),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "initialization_match.json", payload)
    return payload


def integrity_stage() -> dict[str, Any]:
    train_data, valid_data, _audit = load_encoded(valid_subset=64)
    batch = _first_batch(valid_data, torch.device("cpu"), limit=64)
    payload: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION}
    for arm in ("s0", "dense", "dict"):
        model = ARMS[arm](seed=0)
        contract = static_contract_checks(model, batch)
        payload[f"{arm}_contract"] = contract
        if arm in ("dense", "dict"):
            payload[f"{arm}_gradient_viability"] = gradient_viability_checks(model, batch)
    payload["passed"] = bool(
        all(payload[f"{arm}_contract"]["passed"] for arm in ("s0", "dense", "dict"))
    )
    payload["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "integrity_gates.json", payload)
    return payload


def smoke_stage(device: str = "cuda") -> dict[str, Any]:
    payload: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "arms": {}}
    for arm in ("s0", "dense", "dict"):
        summary = train_arm(
            arm,
            seed=0,
            device=device,
            max_epochs=2,
            patience=2,
            train_subset=256,
            valid_subset=128,
            tag=f"smoke_{arm}",
            save_state=False,
            soup=False,
        )
        payload["arms"][arm] = {
            "best_valid_mae": summary["best_valid_mae"],
            "parameters": summary["parameters"],
            "branch_parameters": summary["branch_parameters"],
            "contract_passed": summary["strict_static_contract"]["passed"],
            "gradient_viability": summary["gradient_viability"],
            "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
            "device": summary["device"],
        }
    payload["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "smoke.json", payload)
    return payload


def classify_case(m0: float, md: float, mk: float) -> dict[str, Any]:
    gain_vs_base_dict = float(m0) - float(mk)
    gain_vs_base_dense = float(m0) - float(md)
    dict_specific_gain = float(md) - float(mk)
    if gain_vs_base_dict >= GATE_LOCAL_GAIN and dict_specific_gain >= GATE_DICT_SPECIFIC:
        case = "A_DICT_SPECIFIC_SIGNAL"
    elif gain_vs_base_dict >= GATE_LOCAL_GAIN and dict_specific_gain < GATE_DICT_SPECIFIC:
        case = "B_GENERIC_CAPACITY_SIGNAL"
    elif gain_vs_base_dict < GATE_LOCAL_GAIN:
        case = "C_LOCAL_DICTIONARY_NO_CLEAR_SIGNAL"
    else:  # pragma: no cover - exhaustive
        case = "UNDEFINED"
    return {
        "M0_s0": float(m0),
        "MD_dense": float(md),
        "MK_dict": float(mk),
        "gain_vs_base_dict": gain_vs_base_dict,
        "gain_vs_base_dense": gain_vs_base_dense,
        "dict_specific_gain": dict_specific_gain,
        "gate_local_gain": GATE_LOCAL_GAIN,
        "gate_dict_specific": GATE_DICT_SPECIFIC,
        "case": case,
    }


def analyze_stage() -> dict[str, Any]:
    run_paths = {arm: RUNS_DIR / f"{arm}_seed0.json" for arm in ("s0", "dense", "dict")}
    missing = [str(path) for path in run_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing run outputs: {missing}")
    runs = {arm: _read_json(path) for arm, path in run_paths.items()}
    best = classify_case(
        runs["s0"]["best_valid_mae"],
        runs["dense"]["best_valid_mae"],
        runs["dict"]["best_valid_mae"],
    )
    soup_maes = {
        arm: runs[arm]["soup"].get("soup_valid_mae") for arm in ("s0", "dense", "dict")
    }
    if all(value is not None for value in soup_maes.values()):
        soup_case = classify_case(soup_maes["s0"], soup_maes["dense"], soup_maes["dict"])
    else:
        soup_case = None
    param_audit = parameter_audit()
    init_match = initialization_match()
    total_runs = 3
    wall = sum(float(runs[arm]["wall_clock_s"]) for arm in runs)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "primary_metric": "best official-valid MAE (best checkpoint)",
        "secondary_metric": "fixed Top-5 weight soup official-valid MAE",
        "primary": best,
        "soup": soup_case,
        "gains": {
            "gain_vs_base_dict": best["gain_vs_base_dict"],
            "gain_vs_base_dense": best["gain_vs_base_dense"],
            "dict_specific_gain": best["dict_specific_gain"],
        },
        "runs": {
            arm: {
                "best_valid_mae": runs[arm]["best_valid_mae"],
                "best_epoch": runs[arm]["best_epoch"],
                "epochs_run": runs[arm]["epochs_run"],
                "soup_valid_mae": runs[arm]["soup"].get("soup_valid_mae"),
                "soup_members": runs[arm]["soup"].get("members"),
                "parameters": runs[arm]["parameters"],
                "branch_parameters": runs[arm]["branch_parameters"],
                "wall_clock_s": runs[arm]["wall_clock_s"],
                "peak_gpu_memory_mb": runs[arm].get("peak_gpu_memory_mb"),
                "device": runs[arm]["device"],
                "git_commit": runs[arm]["git_commit"],
                "contract_passed": runs[arm]["strict_static_contract"]["passed"],
                "residual_ablation": runs[arm]["residual_ablation"],
            }
            for arm in ("s0", "dense", "dict")
        },
        "dictionary_diagnostics": runs["dict"]["dictionary_diagnostics"],
        "dictionary_gradient_viability": runs["dict"]["gradient_viability"],
        "dense_gradient_viability": runs["dense"]["gradient_viability"],
        "parameter_audit": param_audit,
        "initialization_match": init_match,
        "budget": {
            "full_training_runs": total_runs,
            "arms": ["S0 seed0", "S-Dense seed0", "S-Dict seed0"],
            "total_wall_clock_s": wall,
            "seed1_purchased": False,
            "hpo_run": False,
            "dictionary_pair_kernel_implemented": False,
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "analysis.json", payload)
    _write_summary_markdown(payload)
    return payload


def _write_summary_markdown(payload: Mapping[str, Any]) -> None:
    primary = payload["primary"]
    soup = payload["soup"]
    runs = payload["runs"]
    diag = payload["dictionary_diagnostics"]
    init_match = payload["initialization_match"]
    lines: list[str] = []
    lines.append("# ZINC strict-static dictionary-pair v0 — results summary")
    lines.append("")
    lines.append(
        "Question: on a modern, strongly-trained **strict-static** patch-pair "
        "baseline, is a residual learnable dictionary more valuable than a "
        "parameter-matched generic dense residual adapter?"
    )
    lines.append("")
    lines.append("Official ZINC **test was never loaded** in any stage.")
    lines.append("")
    lines.append("## Primary metric — best official-valid MAE")
    lines.append("")
    lines.append("| arm | best valid MAE | best epoch | epochs | soup MAE | params | branch params |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm, label in (("s0", "S0"), ("dense", "S-Dense"), ("dict", "S-Dict")):
        row = runs[arm]
        lines.append(
            f"| {label} | {row['best_valid_mae']:.6f} | {row['best_epoch']} | "
            f"{row['epochs_run']} | "
            f"{(row['soup_valid_mae'] if row['soup_valid_mae'] is not None else float('nan')):.6f} | "
            f"{row['parameters']} | {row['branch_parameters']} |"
        )
    lines.append("")
    lines.append("## Pre-registered gains and case")
    lines.append("")
    lines.append(f"* `M0 - MK` (gain_vs_base_dict) = {primary['gain_vs_base_dict']:+.6f}")
    lines.append(f"* `M0 - MD` (gain_vs_base_dense) = {primary['gain_vs_base_dense']:+.6f}")
    lines.append(f"* `MD - MK` (dict_specific_gain) = {primary['dict_specific_gain']:+.6f}")
    lines.append(f"* primary case: **{primary['case']}**")
    if soup is not None:
        lines.append(
            f"* soup case (corroborating): **{soup['case']}** "
            f"(M0 {soup['M0_s0']:.6f}, MD {soup['MD_dense']:.6f}, MK {soup['MK_dict']:.6f})"
        )
    lines.append("")
    lines.append("## Control integrity")
    lines.append("")
    lines.append(f"* strict-static contract passed for every arm: "
                 f"{all(runs[a]['contract_passed'] for a in runs)}")
    lines.append(
        f"* shared tensors bit-identical (S0 vs Dense vs Dict): "
        f"{init_match['bit_identical']} (n={init_match['n_shared_tensors']})"
    )
    lines.append(
        f"* Dense/Dict branch parameter mismatch: "
        f"{payload['parameter_audit']['parameter_match']['relative_error'] * 100:.2f}%"
    )
    lines.append("")
    lines.append("## Dictionary diagnostics (report only; never used for selection)")
    lines.append("")
    for key in (
        "mean_assignment_entropy",
        "effective_prototype_count",
        "active_prototype_count",
        "used_prototype_count_argmax",
        "top8_assignment_mass",
        "max_average_assignment_mass",
        "dictionary_coherence_mean_abs",
        "dictionary_coherence_max_abs",
        "mean_residual_norm",
        "mean_h0_norm",
        "tau_final",
    ):
        lines.append(f"* {key}: {diag.get(key)}")
    lines.append("")
    lines.append("## Inference ablation — dictionary residual forced to zero")
    lines.append("")
    ablation = runs["dict"]["residual_ablation"]
    lines.append(f"* mean |prediction shift|: {ablation.get('mean_abs_prediction_shift')}")
    lines.append(f"* max |prediction shift|: {ablation.get('max_abs_prediction_shift')}")
    lines.append(f"* fraction of predictions shifted > 1e-6: {ablation.get('frac_shift_gt_1e-6')}")
    lines.append("")
    lines.append("## Budget")
    lines.append("")
    budget = payload["budget"]
    lines.append(f"* full training runs: {budget['full_training_runs']} ({', '.join(budget['arms'])})")
    lines.append(f"* total wall clock: {budget['total_wall_clock_s'] / 60.0:.1f} min")
    lines.append("* seed1/2/3: not purchased; HPO: none; dictionary-pair kernel: not implemented")
    lines.append("* official test accessed = false")
    lines.append("")
    (RESULTS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all() -> None:
    parameter_audit()
    initialization_match()
    prepare_encoded()
    integrity_stage()
    analyze_stage()


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
            "smoke",
            "s0",
            "dense",
            "dict",
            "train",
            "analyze",
            "all",
        ],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--seed", type=int, default=0)
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
    elif args.stage == "smoke":
        print(json.dumps(smoke_stage(args.device), indent=2))
    elif args.stage in ("s0", "dense", "dict"):
        summary = train_arm(args.stage, seed=args.seed, device=args.device)
        print(json.dumps({k: v for k, v in summary.items() if "predictions" not in k}, indent=2))
    elif args.stage == "train":
        if args.arm is None:
            raise SystemExit("`train` requires --arm {s0,dense,dict}")
        summary = train_arm(args.arm, seed=args.seed, device=args.device)
        print(json.dumps({k: v for k, v in summary.items() if "predictions" not in k}, indent=2))
    elif args.stage == "analyze":
        print(json.dumps(analyze_stage(), indent=2))
    else:
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
