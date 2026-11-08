"""LCRC-PW-v1: W2 + a one-shot pair-interaction residual.

``h`` is the *single* post-dropout atom state of the frozen W2 trunk (one atom
encode, one bond encode, one ``local`` call).  The branch builds one symmetric
48-dim token per legal unordered atom pair, one layer of 16-query attention
inside each molecule, and one per-molecule residual on the 128 task logits:

    p_ij    = LN(SiLU(W_sum(h_i+h_j) + W_diff|h_i-h_j| + W_prod(h_i*h_j) + E[b]))
    k_ij    = Linear(48,48)(p_ij),   v_ij = Linear(48,48)(p_ij)
    s_kij   = <q_k, k_ij> / sqrt(48)
    a_kij   = softmax_{ij in molecule}(s_kij)
    z_k     = sum_ij a_kij v_ij                       -> z in [G, 16, 48]
    r       = Linear(128,128)(Dropout(SiLU(Linear(768,128)(LN(flatten(z))))))
    logits  = W2_logits + r * 1[n_pairs(molecule) > 0]

The pair token is symmetric in ``i <-> j`` *bitwise* (IEEE addition,
subtraction and multiplication are commutative, ``abs`` is even, and the same
embedding is added).  The last linear is zero-initialised so ``r`` is exactly
zero at step 0; a molecule without any legal pair also gets exactly zero
residual through an explicit mask.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import pw_common as C  # noqa: E402  (sets up the inherited code dirs)
import bs_model as BSM  # noqa: E402


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------


class PairCombiner(nn.Module):
    """``Linear(128,128)(Dropout(0.10)(SiLU(Linear(768,128)(LayerNorm(z)))))``."""

    def __init__(
        self,
        z_dim: int = C.Z_DIM,
        hidden: int = C.COMBINER_HIDDEN,
        n_out: int = C.N_TASKS,
        dropout: float = C.BRANCH_DROPOUT,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(int(z_dim))
        self.fc1 = nn.Linear(int(z_dim), int(hidden))
        self.act = nn.SiLU()
        self.drop = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden), int(n_out))

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(self.norm(z_flat)))))


class PairBranch(nn.Module):
    """One symmetric pair token + one 16-query segment attention per molecule."""

    def __init__(self, width: int, n_out: int):
        super().__init__()
        self.width = int(width)
        self.pair_dim = int(C.PAIR_DIM)
        self.n_queries = int(C.N_QUERIES)
        self.n_out = int(n_out)
        self.w_sum = nn.Linear(self.width, self.pair_dim, bias=False)
        self.w_diff = nn.Linear(self.width, self.pair_dim, bias=False)
        self.w_prod = nn.Linear(self.width, self.pair_dim, bias=False)
        self.dist_embedding = nn.Parameter(torch.empty(C.N_BUCKETS, self.pair_dim))
        self.norm = nn.LayerNorm(self.pair_dim)
        self.query = nn.Parameter(torch.empty(self.n_queries, self.pair_dim))
        self.key = nn.Linear(self.pair_dim, self.pair_dim)
        self.value = nn.Linear(self.pair_dim, self.pair_dim)
        self.combiner = PairCombiner()
        self.scale = float(C.SCORE_SCALE)

    # -- the pair token ----------------------------------------------------

    def pair_tokens(
        self,
        h: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        pair_bucket: torch.Tensor,
    ) -> torch.Tensor:
        hi = h.index_select(0, pair_i)
        hj = h.index_select(0, pair_j)
        # W_sum(h_i + h_j) == W_sum(h_i) + W_sum(h_j); the identity is used so the
        # sum projection is one [N_atoms, 48] matmul instead of a per-pair one.
        sum_term = self.w_sum(hi) + self.w_sum(hj)
        diff_term = self.w_diff((hi - hj).abs())
        prod_term = self.w_prod(hi * hj)
        emb = self.dist_embedding.index_select(0, pair_bucket)
        return self.norm(F.silu(sum_term + diff_term + prod_term + emb))

    # -- one layer of query-wise segment attention -------------------------

    def segment_attention(
        self, token: torch.Tensor, pair_graph: torch.Tensor, n_graphs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(z [G,16,48], attention [P,16], scores [P,16])``.

        The softmax runs over the legal pairs *of one molecule and query*; a
        molecule without a legal pair never receives an entry, so its ``z`` is
        exactly zero (and the residual is masked anyway).
        """
        key = self.key(token)
        value = self.value(token)
        score = (key @ self.query.t()) * self.scale
        if tuple(score.shape) != (token.shape[0], self.n_queries):
            raise RuntimeError(f"score shape {tuple(score.shape)}")
        if token.shape[0] == 0:
            z = token.new_zeros((int(n_graphs), self.n_queries, self.pair_dim))
            return z, token.new_zeros((0, self.n_queries)), score
        neg = torch.finfo(score.dtype).min
        # The shift is a constant (softmax is shift invariant), so the segment
        # max is detached: it is a numerical-stability device, not a term of the
        # gradient.  Both reductions are out-of-place ``scatter_reduce`` /
        # ``index_add`` calls so autograd sees a clean graph.
        seg_max = (
            torch.full((int(n_graphs), self.n_queries), neg, device=score.device, dtype=score.dtype)
            .scatter_reduce(0, pair_graph.unsqueeze(1).expand(-1, self.n_queries), score, reduce="amax",
                            include_self=True)
            .detach()
        )
        weights = torch.exp(score - seg_max.index_select(0, pair_graph))
        denom = torch.zeros_like(seg_max).index_add(0, pair_graph, weights)
        attn = weights / denom.index_select(0, pair_graph)
        # z[:, k, :] = sum_ij attn_ij[k] * value_ij, one scatter per query so the
        # transient tensor is [P, 48] and never [P, 16, 48].
        z = torch.stack(
            [
                value.new_zeros((int(n_graphs), self.pair_dim)).index_add(
                    0, pair_graph, attn[:, k : k + 1] * value
                )
                for k in range(self.n_queries)
            ],
            dim=1,
        )
        return z, attn, score

    # -- forward -----------------------------------------------------------

    def forward(self, batch: Mapping[str, torch.Tensor], h: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        pair_i = batch["pair_i"].to(torch.int64)
        pair_j = batch["pair_j"].to(torch.int64)
        pair_bucket = batch["pair_bucket"].to(torch.int64)
        pair_graph = batch["pair_graph"].to(torch.int64)
        pair_counts = batch["pair_counts"].to(torch.int64)
        n_graphs = int(batch["n_graphs"])
        if int(pair_counts.numel()) != n_graphs:
            raise RuntimeError("pair_counts does not describe every graph in the batch")
        if int(pair_counts.sum().item()) != int(pair_i.numel()):
            raise RuntimeError("pair_counts sum does not match the gathered pair list")
        token = self.pair_tokens(h, pair_i, pair_j, pair_bucket)
        z, attn, score = self.segment_attention(token, pair_graph, n_graphs)
        z_flat = z.reshape(n_graphs, self.n_queries * self.pair_dim)
        residual = self.combiner(z_flat)
        has_pairs = (pair_counts > 0).to(residual.dtype).unsqueeze(1)
        residual = residual * has_pairs
        if tuple(residual.shape) != (n_graphs, self.n_out):
            raise RuntimeError(f"residual logits {tuple(residual.shape)} != {(n_graphs, self.n_out)}")
        with torch.no_grad():
            aux: dict[str, Any] = {
                "residual_logits": residual,
                "n_pairs": int(pair_i.numel()),
                "n_graphs_without_pairs": int((pair_counts == 0).sum().item()),
                "token_rms": float(token.detach().pow(2).mean().sqrt().item()) if token.numel() else 0.0,
                "residual_rms": float(residual.detach().pow(2).mean().sqrt().item()),
                "z_rms": float(z.detach().pow(2).mean().sqrt().item()),
                "score_mean": float(score.detach().mean().item()) if score.numel() else 0.0,
                "attn_max_mean": float(attn.detach().max(dim=0).values.mean().item()) if attn.numel() else 0.0,
            }
        return residual, aux


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class PWModel(BSM.BSModel):
    """Frozen W2 trunk + one one-shot pair-interaction residual."""

    def __init__(self, spec, norm, n_out: int, cfg: Mapping[str, Any]):
        super().__init__(spec, norm, n_out, cfg)
        self.width = int(self.cfg["width"])
        self.branch = PairBranch(self.width, int(n_out))

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_aux: bool = False,
        residual_zero: bool = False,
    ):
        _a, _b, _scal, h = self.encode_h(batch)  # exactly one atom encode, one bond encode, one local, one drop
        base = self.post_local(h, batch)
        if residual_zero:
            if return_aux:
                return base, {"base_logits": base, "residual_logits": torch.zeros_like(base), "h": h}
            return base
        residual, aux = self.branch(batch, h)
        logits = base + residual
        if return_aux:
            out = dict(aux)
            out.update({"base_logits": base, "h": h})
            return logits, out
        return logits


# ---------------------------------------------------------------------------
# initialisation: historical W2 bitwise + private generator for the new tensors
# ---------------------------------------------------------------------------


def _standard_linear_init(linear: nn.Linear, generator: torch.Generator) -> None:
    nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    if linear.bias is not None:
        bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
        linear.bias.uniform_(-bound, bound, generator=generator)


def new_keys(model: nn.Module, base_keys: set[str]) -> list[str]:
    return sorted(k for k in model.state_dict() if k not in base_keys)


def initialize_new_modules(model: PWModel, *, seed: int) -> dict[str, Any]:
    """Initialise only the new tensors; never advances the global torch RNG."""
    before = torch.get_rng_state().clone()
    rng_before = C.rng_state_hashes(C.capture_rng_state())
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSET))
    branch = model.branch
    std = 1.0 / math.sqrt(float(C.PAIR_DIM))
    with torch.no_grad():
        _standard_linear_init(branch.w_sum, gen)
        _standard_linear_init(branch.w_diff, gen)
        _standard_linear_init(branch.w_prod, gen)
        branch.dist_embedding.normal_(0.0, std, generator=gen)
        branch.query.normal_(0.0, std, generator=gen)
        _standard_linear_init(branch.key, gen)
        _standard_linear_init(branch.value, gen)
        _standard_linear_init(branch.combiner.fc1, gen)
        branch.combiner.fc2.weight.zero_()
        branch.combiner.fc2.bias.zero_()
        norm_ok = bool(torch.all(branch.norm.weight == 1).item()) and bool(
            torch.count_nonzero(branch.norm.bias).item() == 0
        )
        combiner_norm_ok = bool(torch.all(branch.combiner.norm.weight == 1).item()) and bool(
            torch.count_nonzero(branch.combiner.norm.bias).item() == 0
        )
        fc2_zero = bool(torch.count_nonzero(branch.combiner.fc2.weight).item() == 0) and bool(
            torch.count_nonzero(branch.combiner.fc2.bias).item() == 0
        )
    if not torch.equal(before, torch.get_rng_state()):
        raise RuntimeError("new-module initialisation changed the global torch RNG state")
    if C.rng_state_hashes(C.capture_rng_state()) != rng_before:
        raise RuntimeError("new-module initialisation changed a tracked RNG state")
    return {
        "kind": "private_generator",
        "private_generator_seed": int(seed) * 1_000_003 + int(C.PRIVATE_GENERATOR_OFFSET),
        "linear": "kaiming_uniform(a=sqrt(5)) + fan_in bias (w_sum/w_diff/w_prod have no bias)",
        "dist_embedding": "normal(0, 1/sqrt(48))",
        "query": "normal(0, 1/sqrt(48))",
        "layer_norm": "default (weight=1, bias=0, no RNG)",
        "combiner.fc2": "weight and bias zeroed (residual logits == 0 at step 0)",
        "fc2_is_zero": bool(fc2_zero),
        "pair_norm_default": bool(norm_ok),
        "combiner_norm_default": bool(combiner_norm_ok),
        "global_rng_unchanged": True,
    }


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("utf-8"))
        h.update(str(tuple(value.shape)).encode("utf-8"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def build_model(
    spec,
    norm: Mapping[str, Any],
    n_out: int = C.N_TASKS,
    *,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    verify_parameters: bool = True,
) -> tuple[PWModel, dict[str, Any]]:
    """Build PW from the frozen, *untrained* historical W2 seed-0 state."""
    base_model, base_audit = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, int(n_out), seed=int(seed), device="cpu")
    cfg = dict(base_model.cfg)
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    base_keys = set(base_state)
    global_rng_before = torch.get_rng_state().clone()

    with torch.random.fork_rng(devices=[]):
        model = PWModel(spec, norm, int(n_out), cfg)

    added = new_keys(model, base_keys)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if set(missing) != set(added) or unexpected:
        raise RuntimeError(
            f"base state load mismatch: missing={sorted(missing)} "
            f"expected_new={sorted(added)} unexpected={sorted(unexpected)}"
        )
    init_audit = initialize_new_modules(model, seed=int(seed))

    subset_diff: dict[str, float] = {}
    state = model.state_dict()
    for k in sorted(base_keys):
        if tuple(state[k].shape) != tuple(base_state[k].shape) or state[k].dtype != base_state[k].dtype:
            raise RuntimeError(f"original W2 tensor shape/dtype changed: {k}")
        if not torch.equal(state[k].cpu(), base_state[k].cpu()):
            subset_diff[k] = float((state[k].float().cpu() - base_state[k].float().cpu()).abs().max().item())
    if subset_diff:
        raise RuntimeError(f"original W2 parameters are not bitwise equal: {subset_diff}")
    if not torch.equal(global_rng_before, torch.get_rng_state()):
        raise RuntimeError("build advanced the global torch RNG state")

    model.to(torch.device(device))
    base_elements = int(sum(p.numel() for n, p in model.named_parameters() if n in base_keys))
    new_elements = int(sum(p.numel() for n, p in model.named_parameters() if n not in base_keys))
    total = int(C.parameter_count(model))
    inventory = {k: {"shape": list(state[k].shape), "numel": int(state[k].numel())} for k in added}
    audit: dict[str, Any] = {
        "model": C.MODEL_KEY,
        "model_name": C.MODEL["name"],
        "role": C.MODEL["role"],
        "seed": int(seed),
        "base_model": C.BASE_MODEL_KEY,
        "base_parameters": base_elements,
        "new_parameters": new_elements,
        "total_parameters": total,
        "expected_base_parameters": int(C.BASE_PARAMETERS),
        "expected_new_parameters": int(C.BRANCH_ADDED),
        "expected_total_parameters": int(C.EXPECTED_PARAMETERS),
        "parameter_ceiling": int(C.PARAMETER_CEILING),
        "widths": cfg,
        "branch_definition": {
            "kind": "one_shot_pair_attention_residual",
            "pair_dim": C.PAIR_DIM,
            "n_queries": C.N_QUERIES,
            "z_dim": C.Z_DIM,
            "combiner_hidden": C.COMBINER_HIDDEN,
            "dropout": C.BRANCH_DROPOUT,
            "distance_buckets": [list(b) for b in C.BUCKETS],
            "score_scale": C.SCORE_SCALE,
            "softmax": "query-wise segment softmax over the legal pairs of one molecule",
            "pair_layers": 1,
            "iterative_message_passing": False,
            "atom_state_writes": 0,
            "pair_state_updates": 0,
            "pair_sampling": False,
            "residual_mask": "exactly zero for molecules with no legal pair",
        },
        "init_rule": C.INIT_RULE,
        "new_module_init": init_audit,
        "new_state_keys": added,
        "new_parameter_inventory": inventory,
        "original_w2_subset": {"n_tensors": len(base_keys), "all_bitwise_equal": True, "n_differing": 0},
        "base_trunk_init_fingerprint": base_audit.get("trunk_init_fingerprint"),
        "base_full_trunk_init_fingerprint": base_audit.get("full_trunk_init_fingerprint"),
        "base_state_sha256": state_sha256(base_state),
        "model_state_sha256": state_sha256(state),
        "state_keys": sorted(state),
        "n_state_tensors": len(state),
        "parameter_tensors": len(list(model.parameters())),
        "base_model_audit": base_audit,
        "global_rng_unchanged": True,
    }
    if verify_parameters:
        if new_elements != int(C.BRANCH_ADDED):
            raise RuntimeError(f"new parameters {new_elements} != {C.BRANCH_ADDED}")
        if total != int(C.EXPECTED_PARAMETERS):
            raise RuntimeError(f"total parameters {total} != {C.EXPECTED_PARAMETERS}")
        if base_elements != int(C.BASE_PARAMETERS):
            raise RuntimeError(f"base parameters {base_elements} != {C.BASE_PARAMETERS}")
        if total > int(C.PARAMETER_CEILING):
            raise RuntimeError(f"{total} parameters exceed the {C.PARAMETER_CEILING} ceiling")
        if norm.get("norm_fingerprint") == C.SOURCE["norm_fingerprint"]:
            if audit["base_state_sha256"] != C.W2["base_state_sha256_at_init"]:
                raise RuntimeError(
                    f"W2 init state {audit['base_state_sha256']} != historical "
                    f"{C.W2['base_state_sha256_at_init']}"
                )
            if audit["base_trunk_init_fingerprint"] != C.W2["trunk_init_fingerprint"]:
                raise RuntimeError("trunk init fingerprint is not the historical seed-0 one")
            audit["historical_init_check"] = "exact"
        else:
            audit["historical_init_check"] = "skipped_non_frozen_norm_fingerprint"
        audit["parameter_check"] = "exact"
    return model, audit


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def _stats(x: torch.Tensor) -> dict[str, Any]:
    x = x.detach().float()
    return {
        "rms": float(x.pow(2).mean().sqrt().item()) if x.numel() else 0.0,
        "mean_abs": float(x.abs().mean().item()) if x.numel() else 0.0,
        "max_abs": float(x.abs().max().item()) if x.numel() else 0.0,
        "l2": float(x.norm().item()) if x.numel() else 0.0,
        "numel": int(x.numel()),
    }


@torch.no_grad()
def branch_diagnostics(model: PWModel, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        logits, aux = model(batch, return_aux=True)
        residual = aux["residual_logits"]
        base = aux["base_logits"]
        return {
            "residual": _stats(residual),
            "base_logits": _stats(base),
            "logits": _stats(logits),
            "h": _stats(aux["h"]),
            "residual_over_base": float(
                residual.pow(2).mean().sqrt().item() / max(base.pow(2).mean().sqrt().item(), 1e-12)
            ),
            "nonzero_fraction": float((residual != 0).float().mean().item()),
            "n_pairs": int(aux["n_pairs"]),
            "n_graphs_without_pairs": int(aux["n_graphs_without_pairs"]),
            "token_rms": float(aux["token_rms"]),
            "z_rms": float(aux["z_rms"]),
            "score_mean": float(aux["score_mean"]),
            "attn_max_mean": float(aux["attn_max_mean"]),
            "pair_count_per_graph": _stats(batch["pair_counts"].detach()),
        }
    finally:
        model.train(was_training)


@torch.no_grad()
def module_norms(model: PWModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if name.startswith("branch."):
            out[name] = {
                "l2": float(param.detach().float().norm().item()),
                "rms": float(param.detach().float().pow(2).mean().sqrt().item()) if param.numel() else 0.0,
                "numel": int(param.numel()),
            }
    return out


@torch.no_grad()
def branch_grad_norm(model: PWModel) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in model.named_parameters():
        if name.startswith("branch.") and param.grad is not None:
            out[name] = float(param.grad.detach().float().norm().item())
    return out


def count_module_calls_typed(model: PWModel, batch: Mapping[str, torch.Tensor]) -> dict[str, int]:
    names = ("atom_encoder", "bond_encoder", "local", "drop", "f", "rho", "readout", "branch")
    counters = {n: 0 for n in names}
    handles = []
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue

        def _hook(_m, _inp, _out, key=name):
            counters[key] += 1

        handles.append(module.register_forward_hook(_hook))
    try:
        model.eval()
        with torch.no_grad():
            model(batch)
    finally:
        for h in handles:
            h.remove()
    return counters


@torch.no_grad()
def attention_entropy(model: PWModel, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Per-query attention entropy of one batch (inference analysis only).

    For a molecule with ``n`` legal pairs, the uniform distribution has entropy
    ``log n``.  Reporting both the raw entropy and ``H / log n`` says whether a
    query is close to uniform (ratio -> 1) or has collapsed onto a few pairs
    (ratio -> 0).  Molecules with a single pair are excluded from the ratio
    (their entropy is 0 by construction and carries no information).
    """
    was_training = model.training
    model.eval()
    try:
        _a, _b, _scal, h = model.encode_h(batch)
        pair_i = batch["pair_i"].to(torch.int64)
        pair_graph = batch["pair_graph"].to(torch.int64)
        pair_counts = batch["pair_counts"].to(torch.int64)
        token = model.branch.pair_tokens(
            h, pair_i, batch["pair_j"].to(torch.int64), batch["pair_bucket"].to(torch.int64)
        )
        _z, attn, score = model.branch.segment_attention(token, pair_graph, int(batch["n_graphs"]))
        if attn.numel() == 0:
            return {"n_pairs": 0, "per_query": []}
        logp = torch.log(attn.clamp_min(1e-30))
        # per (graph, query) entropy: scatter -sum a log a
        ent = torch.zeros(
            (int(batch["n_graphs"]), C.N_QUERIES), device=attn.device, dtype=attn.dtype
        ).index_add(0, pair_graph, -(attn * logp))
        counts = pair_counts.to(attn.dtype)
        per_query = []
        for k in range(C.N_QUERIES):
            hk = ent[:, k]
            logn = torch.log(counts.clamp_min(1.0))
            valid = counts > 1
            ratio = (hk[valid] / logn[valid]) if bool(valid.any()) else torch.zeros(1)
            per_query.append(
                {
                    "query": int(k),
                    "mean_entropy_nats": float(hk.mean().item()),
                    "mean_uniform_entropy_nats": float(logn.mean().item()),
                    "mean_normalised_entropy": float(ratio.mean().item()),
                    "min_normalised_entropy": float(ratio.min().item()),
                    "max_normalised_entropy": float(ratio.max().item()),
                    "mean_max_attention_weight": float(
                        torch.zeros(int(batch["n_graphs"]), device=attn.device)
                        .scatter_reduce(0, pair_graph, attn[:, k], reduce="amax", include_self=True)
                        .mean()
                        .item()
                    ),
                }
            )
        return {
            "n_pairs": int(attn.numel() // C.N_QUERIES),
            "n_graphs": int(batch["n_graphs"]),
            "n_graphs_with_pairs": int((pair_counts > 0).sum().item()),
            "queries": per_query,
            "all_queries_near_uniform": bool(
                all(q["mean_normalised_entropy"] > 0.95 for q in per_query)
            ),
            "mean_normalised_entropy_over_queries": float(
                sum(q["mean_normalised_entropy"] for q in per_query) / len(per_query)
            ),
        }
    finally:
        model.train(was_training)


__all__ = [
    "PWModel",
    "PairBranch",
    "PairCombiner",
    "attention_entropy",
    "branch_diagnostics",
    "branch_grad_norm",
    "build_model",
    "count_module_calls_typed",
    "initialize_new_modules",
    "module_norms",
    "new_keys",
    "state_sha256",
]
