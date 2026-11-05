"""Arms C/F/M/V, strict warm-start loading and AdamW-state restoration.

The frozen C1 class is *subclassed*, not copied: the parametrization and every
original computation (AtomEncoder, BondEncoder, S1/S2 local content, dropout,
`f`, `rho`, global descriptor, normalization) stay the inherited ones.  The new
residual blocks are appended as extra submodules, so the first 31 parameters of
``model.parameters()`` keep exactly the source order and the source AdamW state
can be restored tensor-by-tensor before the new parameter group is appended.

New modules are initialised with a private ``torch.Generator``.  The final
linear layer of every residual block is zero-initialised, so at global step 0
the new residual is exactly zero and each arm reproduces C1 outputs.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sg_common as C
import bp_model as BPM
import pc_model as PC

NEW_MODULE_PREFIXES = ("blocks.", "bond_proj.")
BASE_STATE_TENSORS = 37
BASE_PARAMETERS = 31


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


class ResidualFFN(nn.Module):
    """Pre-norm residual FFN: LayerNorm64 -> Linear 64->128 -> SiLU -> Linear 128->64."""

    def __init__(self, width: int = C.BLOCK_WIDTH, hidden: int = C.BLOCK_HIDDEN):
        super().__init__()
        self.norm = nn.LayerNorm(int(width))
        self.fc1 = nn.Linear(int(width), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        return self.fc2(F.silu(self.fc1(z)))


class MessageFFN(ResidualFFN):
    """Same FFN shape; the normalized input also receives directed bond messages."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        src: torch.Tensor,
        dst: torch.Tensor,
        bond_msg: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.norm(x)
        if src.numel():
            msg = torch.zeros_like(z).index_add_(0, dst, F.silu(z[src] + bond_msg))
            z = z + msg
        if context is not None:
            z = z + context
        return self.fc2(F.silu(self.fc1(z)))


class ContextMessageFFN(MessageFFN):
    """Message block plus a per-graph mean-pool context feedback."""

    def __init__(self, width: int = C.BLOCK_WIDTH, hidden: int = C.BLOCK_HIDDEN):
        super().__init__(width, hidden)
        self.ctx_proj = nn.Linear(int(width), int(width))


# ---------------------------------------------------------------------------
# arm model
# ---------------------------------------------------------------------------


class SGModel(BPM.C1Model):
    """Frozen LCRC-C1 with the arm's residual blocks inserted after dropout."""

    def __init__(self, spec: Any, norm: Mapping[str, Any], n_out: int, arm: str):
        super().__init__(spec, norm, int(n_out))
        if arm not in C.ARMS:
            raise KeyError(arm)
        self.arm = arm
        self.block_mode = C.ARMS[arm]["blocks"]
        self.blocks: nn.ModuleList | None = None
        # Module construction itself draws from the global torch RNG.  Fork it
        # so adding modules cannot shift the public dropout stream; the actual
        # new-module values are then set from a private generator below.
        with torch.random.fork_rng(devices=[]):
            if self.block_mode == "ffn":
                blocks = nn.ModuleList([ResidualFFN() for _ in range(2)])
                bond_proj = None
            elif self.block_mode == "mp":
                blocks = nn.ModuleList([MessageFFN() for _ in range(2)])
                bond_proj = nn.Linear(int(spec.bond_feat_dim), C.BLOCK_WIDTH)
            elif self.block_mode == "mp_context":
                blocks = nn.ModuleList([ContextMessageFFN() for _ in range(2)])
                bond_proj = nn.Linear(int(spec.bond_feat_dim), C.BLOCK_WIDTH)
            elif self.block_mode == "none":
                blocks = None
                bond_proj = None
            else:  # pragma: no cover - defensive
                raise KeyError(self.block_mode)
        self.blocks = blocks
        self.bond_proj = bond_proj

    # -- new-parameter helpers ------------------------------------------------

    def new_parameter_names(self) -> list[str]:
        return [n for n, _ in self.named_parameters() if n.startswith(NEW_MODULE_PREFIXES)]

    def base_parameter_names(self) -> list[str]:
        new = set(self.new_parameter_names())
        return [n for n, _ in self.named_parameters() if n not in new]

    def new_modules(self) -> Iterable[nn.Module]:
        mods: list[nn.Module] = []
        if self.blocks is not None:
            mods.extend(self.blocks)
        if getattr(self, "bond_proj", None) is not None:
            mods.append(self.bond_proj)
        return mods

    def new_parameters(self) -> list[nn.Parameter]:
        names = set(self.new_parameter_names())
        return [p for n, p in self.named_parameters() if n in names]

    def base_parameters(self) -> list[nn.Parameter]:
        names = set(self.new_parameter_names())
        return [p for n, p in self.named_parameters() if n not in names]

    # -- forward --------------------------------------------------------------

    def _bond_directions(self, batch: Mapping[str, torch.Tensor]):
        idx = batch["bond_index"]
        bf = batch["bond_feat"]
        if idx.numel():
            src = torch.cat([idx[:, 0], idx[:, 1]], dim=0)
            dst = torch.cat([idx[:, 1], idx[:, 0]], dim=0)
            bf2 = torch.cat([bf, bf], dim=0)
            bond_msg = self.bond_proj(bf2)
        else:
            empty_i = idx.new_zeros((0,))
            src, dst = empty_i, empty_i
            bond_msg = idx.new_zeros((0, C.BLOCK_WIDTH)).to(bf.dtype)
        return src, dst, bond_msg

    def propagate(self, h: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Run the arm's added branch on atom states ``h`` (no final readout)."""
        if self.blocks is None:
            return h
        if self.block_mode in ("mp", "mp_context"):
            src, dst, bond_msg = self._bond_directions(batch)
        else:
            src = dst = bond_msg = None
        atom_batch = batch["atom_batch"]
        n_graphs = int(batch["n_graphs"])
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        denom = count.clamp(min=1.0).unsqueeze(1)
        for block in self.blocks:
            if self.block_mode == "ffn":
                out = block(h)
            elif self.block_mode == "mp":
                out = block(h, src=src, dst=dst, bond_msg=bond_msg)
            elif self.block_mode == "mp_context":
                z = block.norm(h)
                g = h.new_zeros((n_graphs, h.shape[1])).index_add_(0, atom_batch, z) / denom
                ctx = block.ctx_proj(g)[atom_batch]
                out = block(h, src=src, dst=dst, bond_msg=bond_msg, context=ctx)
            else:  # pragma: no cover - defensive
                raise KeyError(self.block_mode)
            h = h + out
        return h

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        # Identical to ``TULCRC.forward`` except for the inserted arm branch.
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        a, b, scal = self._encode(batch)
        h = self.drop(self.local(batch, a, b, scal))
        h = self.propagate(h, batch)
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
        if return_aux:
            return logits, {"h": h, "contrib": contrib, "rho": rho, "mean_h": mean_h, "mean_h2": mean_h2}
        return logits


# ---------------------------------------------------------------------------
# construction / initialisation
# ---------------------------------------------------------------------------


def build_reference(spec: Any, norm: Mapping[str, Any], n_out: int, seed: int) -> tuple[nn.Module, dict[str, Any], dict[str, torch.Tensor]]:
    """Build the frozen C1 once and return its state dict (CPU)."""
    ref, info = PC.build_c1(spec, norm, int(n_out), int(seed), "cpu")
    if info.get("variant") != "C1" or int(info.get("parameters", -1)) != C.SOURCE["n_parameters"]:
        raise RuntimeError(f"reference C1 is not the frozen 245,184-parameter model: {info.get('parameters')}")
    state = {k: v.detach().cpu().clone() for k, v in ref.state_dict().items()}
    if len(state) != BASE_STATE_TENSORS:
        raise RuntimeError(f"reference state has {len(state)} tensors, expected {BASE_STATE_TENSORS}")
    return ref, info, state


def _new_state_keys(model: SGModel) -> set[str]:
    keys = set(model.state_dict())
    base_keys = {k for k in keys if not k.startswith(NEW_MODULE_PREFIXES)}
    return keys - base_keys


def initialize_new_modules(model: SGModel, *, seed: int, zero_residual: bool = True) -> dict[str, Any]:
    """Private-generator init; global RNG is never touched."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + 17)
    before = torch.get_rng_state().clone()
    n_linear = 0
    n_norm = 0
    with torch.no_grad():
        for module in model.new_modules():
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    bound = 1.0 / math.sqrt(float(sub.in_features))
                    sub.weight.uniform_(-bound, bound, generator=gen)
                    if sub.bias is not None:
                        sub.bias.uniform_(-bound, bound, generator=gen)
                    n_linear += 1
                elif isinstance(sub, nn.LayerNorm):
                    nn.init.ones_(sub.weight)
                    nn.init.zeros_(sub.bias)
                    n_norm += 1
    if zero_residual:
        with torch.no_grad():
            for block in model.blocks or []:
                block.fc2.weight.zero_()
                block.fc2.bias.zero_()
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("private initialisation changed the global torch RNG state")
    return {
        "private_generator_seed": int(seed) * 1_000_003 + 17,
        "linear_layers_initialised": n_linear,
        "layernorm_layers_initialised": n_norm,
        "residual_output_zeroed": bool(zero_residual),
        "global_rng_unchanged": True,
    }


def build_arm(
    arm: str,
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int,
    device: torch.device | str,
    base_init_state: Mapping[str, torch.Tensor],
    warm_start_state: Mapping[str, torch.Tensor] | None,
    zero_residual: bool = True,
) -> tuple[SGModel, dict[str, Any]]:
    """Build one arm and load (i) the frozen C1 init and (ii) optional warm start."""
    model = SGModel(spec, norm, int(n_out), arm)
    missing, unexpected = model.load_state_dict(dict(base_init_state), strict=False)
    expected_missing = _new_state_keys(model)
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"frozen init load mismatch: missing={sorted(missing)} expected_new={sorted(expected_missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    base_fp_init = PC.state_fingerprint(
        {k: v for k, v in model.state_dict().items() if not k.startswith(NEW_MODULE_PREFIXES)}
    )

    warm_audit: dict[str, Any] = {"loaded": False}
    if warm_start_state is not None:
        res = model.load_state_dict(dict(warm_start_state), strict=False)
        if set(res.missing_keys) != expected_missing or res.unexpected_keys:
            raise RuntimeError(
                f"warm-start load mismatch: missing={sorted(res.missing_keys)} "
                f"expected_new={sorted(expected_missing)} unexpected={sorted(res.unexpected_keys)}"
            )
        equality = compare_states(
            {k: v for k, v in model.state_dict().items() if not k.startswith(NEW_MODULE_PREFIXES)},
            warm_start_state,
        )
        if not equality["equal"]:
            raise RuntimeError(f"warm-start base state is not bitwise equal: {equality}")
        warm_audit = {
            "loaded": True,
            "state_dict_bitwise_equal": True,
            "n_tensors_compared": equality["n_tensors"],
            "max_abs_diff": equality["max_abs_diff"],
        }
    base_fp_after_load = PC.state_fingerprint(
        {k: v for k, v in model.state_dict().items() if not k.startswith(NEW_MODULE_PREFIXES)}
    )
    init_audit = initialize_new_modules(model, seed=int(seed), zero_residual=bool(zero_residual))
    model.to(torch.device(device))

    names = model.base_parameter_names()
    if len(names) != BASE_PARAMETERS:
        raise RuntimeError(f"base parameter count {len(names)} != {BASE_PARAMETERS}")
    new_names = model.new_parameter_names()
    base_elements = int(sum(p.numel() for p in model.base_parameters()))
    new_elements = int(sum(p.numel() for p in model.new_parameters()))
    audit = {
        "arm": arm,
        "arm_name": C.ARMS[arm]["name"],
        "block_mode": model.block_mode,
        "base_parameter_count": len(names),
        "base_parameter_elements": base_elements,
        "new_parameter_tensors": len(new_names),
        "new_parameter_count": new_elements,
        "new_parameter_names": new_names,
        "total_parameter_count": int(PC.parameter_count(model)),
        "base_init_fingerprint": base_fp_init,
        "base_state_fingerprint_after_load": base_fp_after_load,
        "warm_start": warm_audit,
        "new_module_init": init_audit,
        "zero_residual_at_init": bool(zero_residual and model.blocks is not None),
        "variant": "C1 + " + model.block_mode,
        "model_info": {
            "parameters": int(PC.parameter_count(model)),
            "base_parameters": len(names),
            "new_parameters": new_elements,
        },
    }
    # The base model info inherited from the frozen build is attached by caller.
    return model, audit


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------


def compare_states(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if set(a) != set(b):
        return {
            "equal": False,
            "reason": "key sets differ",
            "only_a": sorted(set(a) - set(b)),
            "only_b": sorted(set(b) - set(a)),
        }
    max_diff = 0.0
    worst = None
    all_equal = True
    for key in sorted(a):
        av, bv = a[key], b[key]
        if tuple(av.shape) != tuple(bv.shape) or av.dtype != bv.dtype:
            return {
                "equal": False,
                "reason": f"shape/dtype mismatch at {key}",
                "a": [list(av.shape), str(av.dtype)],
                "b": [list(bv.shape), str(bv.dtype)],
            }
        if not torch.equal(av.cpu(), bv.cpu()):
            all_equal = False
            diff = float((av.float().cpu() - bv.float().cpu()).abs().max().item())
            if diff > max_diff:
                max_diff = diff
                worst = key
    return {
        "equal": bool(all_equal),
        "max_abs_diff": float(max_diff),
        "worst_key": worst,
        "n_tensors": len(a),
    }


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    return PC.state_fingerprint(state)


def state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------


def make_optimizer_warm(
    model: SGModel, source_optimizer_state: Mapping[str, Any]
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """Restore the source moments into group 0, then append the new params."""
    base = model.base_parameters()
    new = model.new_parameters()
    if len(base) != BASE_PARAMETERS:
        raise RuntimeError(f"base optimizer parameters {len(base)} != {BASE_PARAMETERS}")
    optimizer = torch.optim.AdamW(
        base,
        lr=float(C.RECIPE["base_lr_start"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    before = optimizer_state_cpu(optimizer)
    if before["state"]:
        raise RuntimeError("fresh optimizer unexpectedly has state")
    optimizer.load_state_dict(copy.deepcopy(dict(source_optimizer_state)))
    loaded = optimizer_state_cpu(optimizer)
    source = {
        "state": {
            int(pid): {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                for k, v in values.items()
            }
            for pid, values in source_optimizer_state["state"].items()
        },
        "param_groups": copy.deepcopy(source_optimizer_state["param_groups"]),
    }
    eq = compare_optimizer_states(loaded, source)
    if not eq["equal"]:
        raise RuntimeError(f"restored AdamW state is not identical to source: {eq}")
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in loaded["state"].values()
        }
    )
    if steps != [float(C.SOURCE["optimizer_step_per_param"])]:
        raise RuntimeError(f"restored AdamW steps {steps} != {C.SOURCE['optimizer_step_per_param']}")
    if len(loaded["param_groups"]) != 1 or list(loaded["param_groups"][0]["params"]) != list(range(BASE_PARAMETERS)):
        raise RuntimeError(f"restored source param groups are not one group 0..30: {loaded['param_groups']}")
    base_fp = optimizer_state_fingerprint(loaded)

    if new:
        optimizer.add_param_group(
            {
                "params": new,
                "lr": float(C.RECIPE["new_lr_start"]),
                "betas": tuple(C.RECIPE["adam_betas"]),
                "eps": float(C.RECIPE["adam_eps"]),
                "weight_decay": float(C.RECIPE["weight_decay"]),
            }
        )
    after = optimizer_state_cpu(optimizer)
    group0_equal = compare_optimizer_states(
        {
            "state": {pid: after["state"][pid] for pid in after["state"] if pid < BASE_PARAMETERS},
            "param_groups": [after["param_groups"][0]],
        },
        loaded,
    )
    if not group0_equal["equal"]:
        raise RuntimeError("appending the new parameter group changed the source moments")
    if new:
        if len(after["param_groups"]) != 2:
            raise RuntimeError("new parameter group was not appended")
        if list(after["param_groups"][1]["params"]) != list(range(BASE_PARAMETERS, BASE_PARAMETERS + len(new))):
            raise RuntimeError("new parameter group ids are not contiguous after the base group")
        if any(pid in after["state"] for pid in range(BASE_PARAMETERS, BASE_PARAMETERS + len(new))):
            raise RuntimeError("new parameters must start without AdamW state")
    audit = {
        "optimizer": "AdamW",
        "source_step_restored": steps,
        "source_optimizer_state_fingerprint": base_fp,
        "group0_identical_to_source": bool(group0_equal["equal"]),
        "moments_reinitialised": False,
        "n_param_groups_after": len(after["param_groups"]),
        "base_param_group_size": len(base),
        "new_param_group_size": len(new),
        "new_param_state_empty_at_start": True,
        "base_lr": float(C.RECIPE["base_lr_start"]),
        "new_lr": float(C.RECIPE["new_lr_start"]),
    }
    return optimizer, audit


def make_optimizer_scratch(model: SGModel) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.SCRATCH_RECIPE["warmup_start_lr"]),
        betas=tuple(C.SCRATCH_RECIPE["adam_betas"]),
        eps=float(C.SCRATCH_RECIPE["adam_eps"]),
        weight_decay=float(C.SCRATCH_RECIPE["weight_decay"]),
    )
    audit = {
        "optimizer": "AdamW",
        "mode": "from_scratch",
        "n_param_groups": len(optimizer.param_groups),
        "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "learning_rate_target": float(C.SCRATCH_RECIPE["learning_rate"]),
        "warmup_start_lr": float(C.SCRATCH_RECIPE["warmup_start_lr"]),
        "moments_reinitialised": True,
    }
    return optimizer, audit


def set_lrs(optimizer: torch.optim.AdamW, *, phase: int, step: int) -> dict[str, float]:
    if int(phase) == 3:
        lr = C.scheduled_lr(step, phase=3, kind="all")
        for g in optimizer.param_groups:
            g["lr"] = lr
        return {"all": float(lr)}
    base_lr = C.scheduled_lr(step, phase=int(phase), kind="base")
    optimizer.param_groups[0]["lr"] = base_lr
    out = {"base": float(base_lr)}
    if len(optimizer.param_groups) > 1:
        new_lr = C.scheduled_lr(step, phase=int(phase), kind="new")
        optimizer.param_groups[1]["lr"] = new_lr
        out["new"] = float(new_lr)
    return out


def optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    out: dict[str, Any] = {"state": {}, "param_groups": copy.deepcopy(state["param_groups"])}
    for param_id, values in state["state"].items():
        out["state"][int(param_id)] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in values.items()
        }
    return out


def _recursive_equal(a: Any, b: Any) -> dict[str, Any]:
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return {"equal": False, "reason": "tensor vs non-tensor"}
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            return {"equal": False, "reason": "shape/dtype mismatch"}
        exact = bool(torch.equal(a, b))
        diff = float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0
        return {"equal": exact, "max_abs_diff": diff}
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return {"equal": False, "reason": "key mismatch"}
        rows = {str(k): _recursive_equal(a[k], b[k]) for k in a}
        return {
            "equal": all(v["equal"] for v in rows.values()),
            "max_abs_diff": max((v.get("max_abs_diff", 0.0) for v in rows.values()), default=0.0),
        }
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return {"equal": False, "reason": "length mismatch"}
        rows = [_recursive_equal(x, y) for x, y in zip(a, b)]
        return {
            "equal": all(v["equal"] for v in rows),
            "max_abs_diff": max((v.get("max_abs_diff", 0.0) for v in rows), default=0.0),
        }
    try:
        same = bool(a == b)
    except Exception:  # pragma: no cover
        same = False
    return {"equal": same}


def compare_optimizer_states(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    return _recursive_equal(a, b)


def _tensor_digest(value: Any) -> str:
    import hashlib

    if torch.is_tensor(value):
        arr = value.detach().cpu().contiguous().numpy()
    else:
        arr = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(tuple(arr.shape)).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def optimizer_state_fingerprint(state: Mapping[str, Any]) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(repr(C._json_default(state.get("param_groups", []))).encode())
    for pid in sorted(state.get("state", {}), key=lambda x: int(x)):
        h.update(str(int(pid)).encode())
        values = state["state"][pid]
        for key in sorted(values):
            h.update(key.encode())
            h.update(_tensor_digest(values[key]).encode())
    return h.hexdigest()


def load_continuation(
    model: SGModel, optimizer: torch.optim.AdamW, payload: Mapping[str, Any], *, arm: str
) -> dict[str, Any]:
    """Strict restore of a Phase-1 epoch-12 checkpoint for Phase 2."""
    if payload.get("arm") != arm:
        raise RuntimeError(f"checkpoint arm {payload.get('arm')} != requested {arm}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError("checkpoint is from another round")
    model.load_state_dict(payload["state_dict"], strict=True)
    saved_opt = payload["optimizer_state_dict"]
    if len(saved_opt.get("param_groups", [])) != len(optimizer.param_groups):
        raise RuntimeError(
            f"checkpoint has {len(saved_opt.get('param_groups', []))} optimizer groups, "
            f"the rebuilt optimizer has {len(optimizer.param_groups)}"
        )
    optimizer.load_state_dict(copy.deepcopy(dict(saved_opt)))
    loaded = optimizer_state_cpu(optimizer)
    eq = compare_optimizer_states(loaded, saved_opt)
    if not eq["equal"]:
        raise RuntimeError(f"restored continuation optimizer state differs: {eq}")
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in loaded["state"].values()
        }
    )
    sched = payload["scheduler_state"]
    consumed = int(payload["global_step"])
    if consumed != int(sched["global_step"]):
        raise RuntimeError("checkpoint global_step and scheduler state disagree")
    return {
        "resumed_from": C.track_rel(payload.get("_path", "")) if payload.get("_path") else "",
        "source_epoch": int(payload.get("epoch", -1)),
        "global_step": consumed,
        "scheduler_state": sched,
        "optimizer_state_identical": True,
        "optimizer_step_values": steps,
        "batch_plan_sha256": payload.get("batch_plan", {}).get("global_plan_sha256"),
        "rng_state_hashes": payload.get("rng_state_hashes"),
    }


def restore_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    import random

    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        states = [s.detach().cpu() for s in state["torch_cuda"]]
        if len(states) != torch.cuda.device_count():
            raise RuntimeError(
                f"RNG state has {len(states)} CUDA devices, host has {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(states)
    return C.rng_state_hashes(C.capture_rng_state())


__all__ = [
    "BASE_PARAMETERS",
    "BASE_STATE_TENSORS",
    "ContextMessageFFN",
    "MessageFFN",
    "NEW_MODULE_PREFIXES",
    "ResidualFFN",
    "SGModel",
    "build_arm",
    "build_reference",
    "compare_optimizer_states",
    "compare_states",
    "initialize_new_modules",
    "load_continuation",
    "make_optimizer_scratch",
    "make_optimizer_warm",
    "optimizer_state_cpu",
    "optimizer_state_fingerprint",
    "restore_rng_state",
    "set_lrs",
    "state_cpu",
    "state_fingerprint",
]
