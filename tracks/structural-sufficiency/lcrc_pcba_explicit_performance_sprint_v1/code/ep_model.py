"""Four fixed models on top of the frozen LCRC-C1 + original A attention readout.

``A0`` is exactly ``NMPModel(arm="A")``: we construct that model with the
historical builder and copy its state into this wrapper unharmed, so its
initialisation, parameter count and forward computation are the frozen ones.

The three additions are zero-residual at initialisation and never read
``bond_index``, ``bond_feat``, ``nbr_*`` or ``path_*``:

* ``A1`` task-conditioned attention over the static atom states ``h``
  (two queries per task, segment softmax inside each molecule only),
* ``A2`` ten standardised RDKit descriptors plus a missing flag,
* ``A3`` offline, deterministic rooted two-hop branch tokens (8192 hashed
  buckets), aggregated as sum+mean over each root atom's out-edges.

``COMBO`` applies the same modules in the frozen order
``A3 -> base readouts -> A1 -> A2`` without any new interaction module.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

import ep_common as C
import nmp_model as NM


WIDTH = 64
N_QUERIES = 2


# ---------------------------------------------------------------------------
# A1: task-conditioned attention over static atom states
# ---------------------------------------------------------------------------


def segment_softmax(scores: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Stable softmax over atoms of the same molecule, per channel."""
    G = int(n_graphs)
    Cc = int(scores.shape[1])
    idx = atom_batch.unsqueeze(1).expand(-1, Cc)
    maxv = scores.new_full((G, Cc), float("-inf"))
    maxv.scatter_reduce_(0, idx, scores, reduce="amax", include_self=False)
    maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
    e = (scores - maxv[atom_batch]).exp()
    den = scores.new_zeros((G, Cc)).index_add_(0, atom_batch, e)
    return e / den[atom_batch].clamp_min(1e-30)


class TaskConditionedReadout(nn.Module):
    """128 tasks x 2 independent queries; graph-level residual per task."""

    def __init__(self, width: int = 64, n_tasks: int = C.N_TASKS, n_queries: int = N_QUERIES,
                 task_chunk: int = 16):
        super().__init__()
        self.width = int(width)
        self.n_tasks = int(n_tasks)
        self.n_queries = int(n_queries)
        self.task_chunk = int(task_chunk)
        self.norm = nn.LayerNorm(self.width)
        self.key_proj = nn.Linear(self.width, self.width)
        self.value_proj = nn.Linear(self.width, self.width)
        self.query = nn.Parameter(torch.empty(self.n_tasks, self.n_queries, self.width))
        self.w = nn.Parameter(torch.zeros(self.n_tasks, self.n_queries, self.width))
        self.b = nn.Parameter(torch.zeros(self.n_tasks))

    def forward(self, h: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        G = int(n_graphs)
        T, K, d = self.n_tasks, self.n_queries, self.width
        z = self.norm(h)
        key = self.key_proj(z)  # (N, d)
        value = self.value_proj(z)  # (N, d)
        scores = torch.einsum("nd,tkd->ntk", key, self.query) / math.sqrt(float(d))
        alpha = segment_softmax(scores.reshape(-1, T * K), atom_batch, G)  # (N, T*K)
        out = h.new_zeros((G, T))
        for start in range(0, T, self.task_chunk):
            end = min(T, start + self.task_chunk)
            chunk = end - start
            a = alpha[:, start * K : end * K]  # (N, chunk*K)
            prod = a.unsqueeze(2) * value.unsqueeze(1)  # (N, chunk*K, d)
            zc = h.new_zeros((G, chunk * K, d)).index_add_(0, atom_batch, prod)
            wc = self.w[start:end].reshape(chunk * K, d)
            out[:, start:end] = (zc * wc).sum(dim=2).view(G, chunk, K).sum(dim=2) + self.b[start:end]
        return out


# ---------------------------------------------------------------------------
# A2: physicochemical descriptor residual
# ---------------------------------------------------------------------------


class PhyschemResidual(nn.Module):
    """Linear(11,64) -> SiLU -> Dropout(0.10) -> Linear(64,128), zero-output."""

    def __init__(self, in_dim: int = C.PHYSCHEM_DIM, hidden: int = 64, n_out: int = C.N_TASKS,
                 dropout: float = 0.10):
        super().__init__()
        self.fc1 = nn.Linear(int(in_dim), int(hidden))
        self.drop = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden), int(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(torch.nn.functional.silu(self.fc1(x))))


# ---------------------------------------------------------------------------
# A3: explicit rooted branch-token residual
# ---------------------------------------------------------------------------


class BranchTokenResidual(nn.Module):
    """Embed(8192,32) -> per-root-atom (sum,mean) -> Linear(64,64), zero-output."""

    def __init__(self, n_buckets: int = C.N_BRANCH_BUCKETS, dim: int = C.BRANCH_TOKEN_DIM,
                 width: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.embed = nn.Embedding(int(n_buckets), self.dim)
        self.proj = nn.Linear(2 * self.dim, int(width))

    def forward(self, h: torch.Tensor, bt_ptr: torch.Tensor, bt_bucket: torch.Tensor) -> torch.Tensor:
        n_atoms = h.shape[0]
        counts = bt_ptr[1:] - bt_ptr[:-1]  # (N_atoms,)
        if bt_bucket.numel():
            atom_of = torch.repeat_interleave(
                torch.arange(n_atoms, device=h.device, dtype=torch.long), counts
            )
            emb = self.embed(bt_bucket.long())
            total = h.new_zeros((n_atoms, self.dim)).index_add_(0, atom_of, emb)
        else:
            total = h.new_zeros((n_atoms, self.dim))
        mean = total / counts.clamp(min=1).unsqueeze(1).to(total.dtype)
        return self.proj(torch.cat([total, mean], dim=1))


# ---------------------------------------------------------------------------
# composed model
# ---------------------------------------------------------------------------

EXTRA_PREFIXES = ("tc.", "pc.", "bt.")
EXTRA_ATTRS = {"A1": "tc", "A2": "pc", "A3": "bt"}


class EPModel(NM.NMPModel):
    """Original A plus a fixed subset of the three zero-residual modules."""

    def __init__(self, spec: Any, norm: Mapping[str, Any], n_out: int, modules: Sequence[str] = ()):
        mods = tuple(modules)
        for m in mods:
            if m not in EXTRA_ATTRS:
                raise KeyError(m)
        # Construction draws are discarded by the private initialiser; fork the
        # global RNG so building a model never advances the dropout stream.
        with torch.random.fork_rng(devices=[]):
            super().__init__(spec, norm, int(n_out), arm="A")
            self.ep_modules = mods
            if "A1" in mods:
                self.tc: nn.Module = TaskConditionedReadout(width=WIDTH, n_tasks=int(n_out))
            if "A2" in mods:
                self.pc: nn.Module = PhyschemResidual(n_out=int(n_out))
            if "A3" in mods:
                self.bt: nn.Module = BranchTokenResidual(width=WIDTH)

    # -- parameter helpers -------------------------------------------------

    def extra_parameter_names(self) -> list[str]:
        return [n for n, _ in self.named_parameters() if n.startswith(EXTRA_PREFIXES)]

    def extra_modules_list(self) -> list[nn.Module]:
        return [getattr(self, EXTRA_ATTRS[m]) for m in self.ep_modules]

    def extra_parameters(self) -> list[nn.Parameter]:
        names = set(self.extra_parameter_names())
        return [p for n, p in self.named_parameters() if n in names]

    def base_parameters(self) -> list[nn.Parameter]:
        names = set(self.extra_parameter_names())
        return [p for n, p in self.named_parameters() if n not in names]

    def extra_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.extra_parameters()))

    # -- forward -----------------------------------------------------------

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        if not self.ep_modules:
            return super().forward(batch, return_aux=return_aux)

        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        a, b, scal = self._encode(batch)
        h = self.drop(self.local(batch, a, b, scal))
        delta_h = None
        if "A3" in self.ep_modules:
            delta_h = self.bt(h, batch["bt_ptr"], batch["bt_bucket"])
            h = h + delta_h
        want_attention = bool(return_aux and self.mode == "attention_pool")
        h, attention, pooled, residual = self.propagate(h, batch, return_attention=want_attention)
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib = h.new_zeros(n_graphs, self.n_classes).index_add_(0, atom_batch, self.f(h))
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in)
        logits = contrib + rho
        if residual is not None:
            logits = logits + residual
        tc_residual = pc_residual = None
        if "A1" in self.ep_modules:
            tc_residual = self.tc(h, atom_batch, n_graphs)
            logits = logits + tc_residual
        if "A2" in self.ep_modules:
            pc_residual = self.pc(batch["physchem"])
            logits = logits + pc_residual
        if return_aux:
            return logits, {
                "h": h,
                "h_delta_branch": delta_h,
                "contrib": contrib,
                "rho": rho,
                "mean_h": mean_h,
                "mean_h2": mean_h2,
                "attention_weights": attention,
                "attention_pooled": pooled,
                "residual_logits": residual,
                "task_conditioned_residual": tc_residual,
                "physchem_residual": pc_residual,
            }
        return logits


# ---------------------------------------------------------------------------
# construction / private initialisation
# ---------------------------------------------------------------------------


MODULE_INIT_OFFSET = {"A1": 101, "A2": 211, "A3": 307}


def module_init_seed(seed: int, module: str) -> int:
    """Per-module private generator seed: independent of the other modules."""
    if module not in MODULE_INIT_OFFSET:
        raise KeyError(module)
    return int(seed) * 1_000_003 + MODULE_INIT_OFFSET[module]


def _extra_init_audit(model: EPModel, seed: int) -> dict[str, Any]:
    """Initialise each added module with its own private generator.

    Using one generator per module (rather than one shared stream) guarantees
    that a module's initial tensors are identical in the single model and in any
    COMBO that contains it.
    """
    before = torch.get_rng_state().clone()
    per_module: dict[str, Any] = {}
    with torch.no_grad():
        for name in ("A1", "A2", "A3"):
            if name not in model.ep_modules:
                continue
            gen_seed = module_init_seed(seed, name)
            gen = torch.Generator(device="cpu").manual_seed(int(gen_seed))
            module = getattr(model, EXTRA_ATTRS[name])
            counts = {"linear": 0, "layernorm": 0, "embedding": 0, "query": 0}
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    bound = 1.0 / math.sqrt(float(sub.in_features))
                    sub.weight.uniform_(-bound, bound, generator=gen)
                    if sub.bias is not None:
                        sub.bias.uniform_(-bound, bound, generator=gen)
                    counts["linear"] += 1
                elif isinstance(sub, nn.LayerNorm):
                    nn.init.ones_(sub.weight)
                    nn.init.zeros_(sub.bias)
                    counts["layernorm"] += 1
                elif isinstance(sub, nn.Embedding):
                    sub.weight.normal_(0.0, 1.0, generator=gen)
                    counts["embedding"] += 1
            if name == "A1":
                module.query.normal_(0.0, 1.0, generator=gen)
                module.w.zero_()
                module.b.zero_()
                counts["query"] = 1
            elif name == "A2":
                module.fc2.weight.zero_()
                module.fc2.bias.zero_()
            elif name == "A3":
                module.proj.weight.zero_()
                module.proj.bias.zero_()
            per_module[name] = {"private_generator_seed": int(gen_seed), "initialised": counts}
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("extra-module initialisation changed the global torch RNG state")
    return {
        "per_module": per_module,
        "global_rng_unchanged": True,
        "zero_residual": {
            "A1_w_and_b": bool("A1" not in model.ep_modules or float(model.tc.w.abs().sum()) == 0.0),
            "A2_last_linear": bool(
                "A2" not in model.ep_modules
                or (float(model.pc.fc2.weight.abs().sum()) == 0.0 and float(model.pc.fc2.bias.abs().sum()) == 0.0)
            ),
            "A3_projection": bool(
                "A3" not in model.ep_modules
                or (float(model.bt.proj.weight.abs().sum()) == 0.0 and float(model.bt.proj.bias.abs().sum()) == 0.0)
            ),
        },
        "query_std": float(model.tc.query.std().item()) if "A1" in model.ep_modules else None,
    }


def build_model(
    arm: str,
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int,
    device: torch.device | str,
    base_init_state: Mapping[str, torch.Tensor],
    combine: str = "",
    zero_residual: bool = True,
) -> tuple[EPModel, dict[str, Any]]:
    """Build one arm with the frozen A initialisation plus private new modules."""
    modules = C.arm_modules(arm, combine)
    # The historical builder produces the exact A initialisation (C1 state +
    # the original 4-head readout drawn from the same private generator).
    base_model, base_audit = NM.build_arm(
        "A",
        spec,
        norm,
        int(n_out),
        seed=int(seed),
        device="cpu",
        base_init_state=base_init_state,
        zero_residual=True,
    )
    model = EPModel(spec, norm, int(n_out), modules)
    missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
    expected_missing = {n for n in model.state_dict() if n.startswith(EXTRA_PREFIXES)}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"frozen A init load mismatch: missing={sorted(missing)} expected={sorted(expected_missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    base_fp_before = C.state_fingerprint(
        {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith(EXTRA_PREFIXES) and not k.startswith(("blocks.", "readout."))
        }
    )
    if base_fp_before != C.HISTORICAL_A0["base_init_fingerprint"]:
        raise RuntimeError(f"A initialisation fingerprint {base_fp_before} != historical seed-0 fingerprint")
    # Original new-module init (the 4-head readout) uses the historical private
    # generator; extras use a second private generator so the readout is untouched.
    nm_audit = NM.initialize_new_modules(model, seed=int(seed), zero_residual=True)
    extra_audit = _extra_init_audit(model, int(seed))
    if not zero_residual:
        raise RuntimeError("this round only supports the zero-residual initialisation")
    base_fp_after = C.state_fingerprint(
        {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith(EXTRA_PREFIXES) and not k.startswith(("blocks.", "readout."))
        }
    )
    if base_fp_after != base_fp_before:
        raise RuntimeError("new-module initialisation changed the frozen A state")
    # The original readout must match the historical A model tensor-for-tensor.
    for name, value in base_model.state_dict().items():
        if not name.startswith(EXTRA_PREFIXES) and not torch.equal(value, model.state_dict()[name].cpu()):
            raise RuntimeError(f"original A tensor {name} was modified by the extended model")
    expected_total = C.expected_parameters(modules)
    base_elements = int(sum(p.numel() for p in model.base_parameters()))
    extra_elements = int(model.extra_parameter_count())
    total = int(sum(p.numel() for p in model.parameters()))
    if base_elements != C.A0_PARAMETERS or total != expected_total or base_elements + extra_elements != total:
        raise RuntimeError(
            f"arm {arm}{modules}: parameters {total} (base {base_elements}, extra {extra_elements}) "
            f"!= expected {expected_total}"
        )
    if extra_elements != sum(int(C.MODULES[m]["new_parameters"]) for m in modules):
        raise RuntimeError("extra parameter count differs from the frozen module table")
    model.to(torch.device(device))
    audit = {
        "arm": arm,
        "arm_name": C.arm_label(arm, modules),
        "modules": list(modules),
        "module_names": [C.MODULES[m]["name"] for m in modules],
        "base_parameters": base_elements,
        "extra_parameters": extra_elements,
        "total_parameters": total,
        "base_init_fingerprint": base_fp_before,
        "base_state_fingerprint_after_init": base_fp_after,
        "historical_a_state_match": True,
        "original_a_audit": {
            "new_parameter_count": int(base_audit["new_parameter_count"]),
            "total_parameter_count": int(base_audit["total_parameter_count"]),
            "new_module_init": base_audit["new_module_init"],
        },
        "original_init": nm_audit,
        "extra_init": extra_audit,
        "module_spec_sha256": C.digest_module_spec(modules),
        "reads": {m: list(C.MODULES[m]["reads"]) for m in modules},
    }
    return model, audit


def state_fingerprint_of(module: nn.Module, *, exclude_prefixes: Iterable[str] = ()) -> str:
    skip = tuple(exclude_prefixes)
    state = {k: v.detach().cpu() for k, v in module.state_dict().items() if not k.startswith(skip)}
    return C.state_fingerprint(state)


__all__ = [
    "BranchTokenResidual",
    "EPModel",
    "N_QUERIES",
    "PhyschemResidual",
    "TaskConditionedReadout",
    "build_model",
    "module_init_seed",
    "segment_softmax",
    "state_fingerprint_of",
]
