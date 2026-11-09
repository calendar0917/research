"""W2-DYNAMIC: the frozen LCRC-W2 plus two directed-bond message FFN layers.

Everything of W2 is inherited, not copied::

    W2DynamicModel(BSModel)   # bs_model.BSModel == LCRC-W2 at scale 2
        ._encode / .local / .drop / .f / .rho / .readout / .global_*  (unchanged)
        .post_local(h2, batch)                                     (unchanged)
        + .bond_proj : Linear(13, 128)                             (new, shared)
        + .blocks    : [MessageFFN(128, 256), MessageFFN(128, 256)] (new, 2 layers)

Forward::

    a, b, scal = self._encode(batch)
    h0 = self.drop(self.local(batch, a, b, scal))       # one encode, one local
    h1 = h0 + blocks[0](h0, src, dst, edge_msg)
    h2 = h1 + blocks[1](h1, src, dst, edge_msg)         # reads h1, never h0
    logits = self.post_local(h2, batch)

``blocks[*]`` are the historical ``sg_model.MessageFFN`` objects with explicit
widths (128 / 256), so the validated pure-bond message formula
``z = LayerNorm(h); z = z + sum_in SiLU(z[src] + edge_msg); fc2(SiLU(fc1(z)))``
is reused verbatim -- no attention, no graph context, no degree normalisation,
no extra pooling or head.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import dy_common as C
import bs_model as BM
from sg_model import MessageFFN


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class W2DynamicModel(BM.BSModel):
    """Frozen W2 computation with two inserted dynamic message layers."""

    def __init__(
        self,
        spec: Any,
        norm: Mapping[str, Any],
        n_out: int,
        cfg: Mapping[str, Any],
        *,
        n_layers: int = 2,
        block_hidden: int = 256,
    ):
        super().__init__(spec, norm, int(n_out), dict(cfg))
        width = int(self.cfg["width"])
        if int(n_layers) != int(C.DYNAMIC["n_layers"]):
            raise ValueError(f"this round fixes exactly {C.DYNAMIC['n_layers']} dynamic layers")
        if int(block_hidden) != width * 2:
            raise ValueError("dynamic block hidden width is 2 * width")
        self.n_dynamic_layers = int(n_layers)
        self.bond_proj = nn.Linear(int(spec.bond_feat_dim), width)
        self.blocks = nn.ModuleList(
            [MessageFFN(width=width, hidden=int(block_hidden)) for _ in range(int(n_layers))]
        )

    # -- parameter grouping -------------------------------------------------

    def new_parameter_names(self) -> list[str]:
        return [n for n, _ in self.named_parameters() if C.is_new_parameter(n)]

    def base_parameter_names(self) -> list[str]:
        new = set(self.new_parameter_names())
        return [n for n, _ in self.named_parameters() if n not in new]

    def new_modules(self) -> list[nn.Module]:
        return [*self.blocks, self.bond_proj]

    def new_parameters(self) -> list[nn.Parameter]:
        names = set(self.new_parameter_names())
        return [p for n, p in self.named_parameters() if n in names]

    def base_parameters(self) -> list[nn.Parameter]:
        names = set(self.new_parameter_names())
        return [p for n, p in self.named_parameters() if n not in names]

    # -- directed real chemical bonds ---------------------------------------

    def bond_directions(
        self,
        batch: Mapping[str, Any],
        *,
        message: str = "real",
        shuffle_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(src, dst, edge_msg)`` for both directions of every physical bond.

        ``message`` selects the *diagnostic* intervention used after training:

        * ``"real"`` -- the real correspondence (the only mode used in training);
        * ``"off"``  -- empty edge list, so ``MessageFFN``'s message term is zero;
        * ``"shuffle"`` -- within each molecule only the ``src`` assignment is
          permuted (``dst`` multiplicity, ``h0`` and all W2 static inputs are
          untouched).
        """
        if message not in ("real", "off", "shuffle"):
            raise KeyError(message)
        idx = batch["bond_index"]
        bf = batch["bond_feat"]
        width = int(self.cfg["width"])
        if idx.numel() and message != "off":
            src = torch.cat([idx[:, 0], idx[:, 1]], dim=0)
            dst = torch.cat([idx[:, 1], idx[:, 0]], dim=0)
            edge_msg = self.bond_proj(torch.cat([bf, bf], dim=0))
            if message == "shuffle":
                src = shuffle_src_within_molecule(
                    src,
                    dst,
                    batch["atom_batch"],
                    seed=int(C.MECHANISM["shuffle_seed"] if shuffle_seed is None else shuffle_seed),
                )
            return src, dst, edge_msg
        empty = idx.new_zeros((0,))
        return empty, empty, bf.new_zeros((0, width))

    # -- the dynamic stack --------------------------------------------------

    def dynamic_layers(
        self,
        h0: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        message: str = "real",
        shuffle_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(h1, h2)``; layer 2 reads ``h1`` (never ``h0``)."""
        src, dst, edge_msg = self.bond_directions(batch, message=message, shuffle_seed=shuffle_seed)
        h1 = h0 + self.blocks[0](h0, src=src, dst=dst, bond_msg=edge_msg)
        h2 = h1 + self.blocks[1](h1, src=src, dst=dst, bond_msg=edge_msg)
        return h1, h2

    def encode_h0(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exactly one atom/bond encode and one ``local`` + one ``drop`` call."""
        a, b, scal = self._encode(batch)
        return a, b, scal, self.drop(self.local(batch, a, b, scal))

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_aux: bool = False,
        return_attention: bool = False,
        message: str = "real",
        shuffle_seed: int | None = None,
    ):
        a, b, scal, h0 = self.encode_h0(batch)
        h1, h2 = self.dynamic_layers(h0, batch, message=message, shuffle_seed=shuffle_seed)
        if not return_aux:
            return self.post_local(h2, batch)
        logits, aux = self.post_local(h2, batch, return_aux=True, return_attention=return_attention)
        aux.update({"h0": h0, "h1": h1, "h2": h2, "a": a, "b": b, "scal": scal})
        return logits, aux


# ---------------------------------------------------------------------------
# message-source shuffle (diagnostic only)
# ---------------------------------------------------------------------------


def shuffle_src_within_molecule(
    src: torch.Tensor,
    dst: torch.Tensor,
    atom_batch: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Permute ``src`` inside every molecule, keeping ``dst`` (and its counts) fixed."""
    if src.numel() == 0:
        return src
    mol = atom_batch.index_select(0, dst).detach().cpu().numpy()
    src_cpu = src.detach().cpu().numpy()
    out = src_cpu.copy()
    rng = np.random.default_rng(int(seed))
    for g in np.unique(mol):
        pos = np.flatnonzero(mol == g)
        out[pos] = src_cpu[pos][rng.permutation(pos.size)]
    if not np.array_equal(np.sort(out), np.sort(src_cpu)):
        raise RuntimeError("shuffle changed the multiset of source atoms")
    if not np.array_equal(mol, atom_batch.index_select(0, dst).detach().cpu().numpy()):
        raise RuntimeError("shuffle changed the destination side")
    return torch.as_tensor(out, device=src.device, dtype=src.dtype)


def check_edges_intra_molecule(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Every dynamic edge stays inside one molecule (no cross-molecule read)."""
    idx = batch["bond_index"]
    atom_batch = batch["atom_batch"]
    if idx.numel() == 0:
        return {"n_bonds": 0, "checked": True, "cross_molecule_edges": 0}
    src = torch.cat([idx[:, 0], idx[:, 1]])
    dst = torch.cat([idx[:, 1], idx[:, 0]])
    same = atom_batch.index_select(0, src) == atom_batch.index_select(0, dst)
    return {
        "n_bonds": int(idx.shape[0]),
        "n_directed": int(src.numel()),
        "cross_molecule_edges": int((~same).sum()),
        "checked": bool(bool(same.all())),
        "both_directions": bool(src.numel() == 2 * idx.shape[0]),
        "each_physical_bond_twice": bool(
            torch.equal(
                torch.sort(src).values,
                torch.sort(torch.cat([idx[:, 0], idx[:, 1]])).values,
            )
        ),
    }


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


def initialize_dynamic_modules(model: W2DynamicModel, *, seed: int, zero_residual: bool = True) -> dict[str, Any]:
    """Private-generator init of the new modules; the global RNG is never advanced."""
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
            for block in model.blocks:
                block.fc2.weight.zero_()
                block.fc2.bias.zero_()
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("private initialisation changed the global torch RNG state")
    return {
        "private_generator_seed": int(seed) * 1_000_003 + 17,
        "linear_layers_initialised": n_linear,
        "layernorm_layers_initialised": n_norm,
        "expected_linear_layers": 1 + 2 * 2,
        "expected_layernorm_layers": 2,
        "residual_output_zeroed": bool(zero_residual),
        "global_rng_unchanged": True,
    }


# ---------------------------------------------------------------------------
# parameter audit
# ---------------------------------------------------------------------------


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    """Per-tensor parameter audit against the frozen 1,128,468 budget."""
    per_tensor: dict[str, dict[str, Any]] = {}
    groups = {"w2_base": 0, "w2_readout": 0, "bond_proj": 0, "blocks.0": 0, "blocks.1": 0}
    for name, p in model.named_parameters():
        n = int(p.numel())
        per_tensor[name] = {"shape": list(p.shape), "numel": n, "new": C.is_new_parameter(name)}
        if name.startswith("bond_proj."):
            groups["bond_proj"] += n
        elif name.startswith("blocks.0."):
            groups["blocks.0"] += n
        elif name.startswith("blocks.1."):
            groups["blocks.1"] += n
        elif name.startswith("readout."):
            groups["w2_readout"] += n
        else:
            groups["w2_base"] += n
    new_total = groups["bond_proj"] + groups["blocks.0"] + groups["blocks.1"]
    w2_total = groups["w2_base"] + groups["w2_readout"]
    total = w2_total + new_total
    expected = C.EXPECTED_PARAMETERS
    checks = {
        "w2_base": groups["w2_base"] == expected["w2_base"],
        "w2_readout": groups["w2_readout"] == expected["w2_readout"],
        "w2_total": w2_total == expected["w2_total"],
        "bond_proj": groups["bond_proj"] == expected["bond_proj"],
        "message_ffn_one": groups["blocks.0"] == expected["message_ffn_one"],
        "message_ffn_two": groups["blocks.0"] + groups["blocks.1"] == expected["message_ffn_two"],
        "new_total": new_total == expected["new_total"],
        "total": total == expected["total"],
    }
    return {
        "groups": groups,
        "new_total": int(new_total),
        "w2_total": int(w2_total),
        "total": int(total),
        "expected": dict(expected),
        "checks": checks,
        "all_match": bool(all(checks.values())),
        "per_tensor": per_tensor,
        "n_parameter_tensors": len(per_tensor),
        "n_state_tensors": len(model.state_dict()),
    }


def block_fingerprints(model: nn.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, module in (
        ("bond_proj", model.bond_proj),
        ("blocks.0", model.blocks[0]),
        ("blocks.1", model.blocks[1]),
    ):
        out[name] = C.state_fingerprint({k: v for k, v in module.state_dict().items()})
    return out


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def build_dynamic_model(
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    zero_residual: bool = True,
    verify: bool = True,
) -> tuple[W2DynamicModel, dict[str, Any]]:
    """Build ``W2-DYNAMIC`` from the official W2 seed-0 constructor state.

    The W2 part is taken *from* :func:`bs_model.build_model` (the official
    constructor, audited for 994,324 parameters), so no W2 initialisation rule is
    re-derived here.  The new modules are constructed inside a forked RNG and
    their values then come from the private generator.
    """
    cfg = BC_config()
    reference, ref_audit = BM.build_model(
        C.W2_KEY, spec, norm, int(n_out), seed=int(seed), device="cpu", verify_parameters=True
    )
    rng_after_w2 = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = W2DynamicModel(spec, norm, int(n_out), cfg)
    if not torch.equal(rng_after_w2, torch.get_rng_state()):
        raise RuntimeError("W2-DYNAMIC construction advanced the global torch RNG")
    load = model.load_state_dict(reference.state_dict(), strict=False)
    expected_missing = {k for k in model.state_dict() if C.is_new_parameter(k)}
    if set(load.missing_keys) != expected_missing or load.unexpected_keys:
        raise RuntimeError(
            f"W2 state load mismatch: missing={sorted(load.missing_keys)} "
            f"expected_new={sorted(expected_missing)} unexpected={sorted(load.unexpected_keys)}"
        )
    base_before = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    init_audit = initialize_dynamic_modules(model, seed=int(seed), zero_residual=bool(zero_residual))
    base_after = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    if base_before != base_after:
        raise RuntimeError("dynamic initialisation changed the W2 part of the state")
    model.to(torch.device(device))

    audit = parameter_audit(model)
    if verify and not audit["all_match"]:
        raise RuntimeError(f"W2-DYNAMIC parameter audit failed: {audit['checks']}")
    ref_state = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
    ours_base = C.state_subset({k: v.detach().cpu() for k, v in model.state_dict().items()}, new=False)
    base_equal = compare_states(ours_base, ref_state)
    if verify and not base_equal["all_equal"]:
        raise RuntimeError(f"W2 part is not bitwise identical to the W2 seed-0 constructor: {base_equal}")
    out = {
        "model": C.MODEL_NAME,
        "base_model": C.W2_KEY,
        "seed": int(seed),
        "widths": dict(cfg),
        "dynamic": dict(C.DYNAMIC),
        "parameters": audit,
        "w2_reference_parameters": int(ref_audit["total_parameters"]),
        "w2_reference_expected_parameters": int(ref_audit["expected_parameters"]),
        "w2_part_bitwise_equal_reference": base_equal,
        "w2_trunk_init_fingerprint": base_before,
        "expected_w2_trunk_init_fingerprint": C.W2_SEED0["trunk_init_fingerprint"],
        "w2_trunk_fingerprint_matches_recorded_w2": base_before
        == C.W2_SEED0["trunk_init_fingerprint"],
        "init": init_audit,
        "block_fingerprints": block_fingerprints(model),
        "state_keys": sorted(model.state_dict()),
        "n_state_tensors": len(model.state_dict()),
        "base_state_sha256": base_before,
        "dynamic_state_sha256": C.state_fingerprint(C.state_subset(model.state_dict(), new=True)),
        "device": str(device),
    }
    if verify and not out["w2_trunk_fingerprint_matches_recorded_w2"]:
        raise RuntimeError(
            f"W2 trunk fingerprint {base_before} != recorded W2 seed-0 "
            f"{C.W2_SEED0['trunk_init_fingerprint']}"
        )
    return model, out


def BC_config() -> dict[str, Any]:
    return dict(C.config_for_scale(int(C.MODELS[C.W2_KEY]["scale"])))


def untrained_w2(
    spec: Any, norm: Mapping[str, Any], n_out: int, *, seed: int = C.SEED, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """The W2 seed-0 model with *no* supervised training (audit reference)."""
    return BM.build_model(C.W2_KEY, spec, norm, int(n_out), seed=int(seed), device=device)


def compare_states(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Per-tensor bitwise comparison of two state dicts with identical keys."""
    if set(a) != set(b):
        return {
            "all_equal": False,
            "reason": "key mismatch",
            "missing_in_a": sorted(set(b) - set(a)),
            "missing_in_b": sorted(set(a) - set(b)),
            "n_tensors": 0,
        }
    diffs = {}
    for key in a:
        ta = torch.as_tensor(a[key])
        tb = torch.as_tensor(b[key])
        diffs[key] = {
            "equal": bool(ta.shape == tb.shape and torch.equal(ta, tb)),
            "max_abs_diff": float((ta.float() - tb.float()).abs().max().item()) if ta.numel() else 0.0,
        }
    return {
        "all_equal": all(v["equal"] for v in diffs.values()),
        "n_tensors": len(diffs),
        "n_unequal": sum(1 for v in diffs.values() if not v["equal"]),
        "per_tensor": diffs,
    }


# ---------------------------------------------------------------------------
# dynamic-usage diagnostics (RMS of the two updates)
# ---------------------------------------------------------------------------


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item()) if x.numel() else 0.0


def usage_metrics(model: W2DynamicModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    """``R1 / R2 / Rtotal`` plus fc2 weight norms and finiteness (eval mode)."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            _a, _b, _scal, h0 = model.encode_h0(batch)
            h1, h2 = model.dynamic_layers(h0, batch)
            d1 = h1 - h0
            d2 = h2 - h1
            dt = h2 - h0
            r0, r1, r2 = rms(h0), rms(h1), rms(h2)
            delta1, delta2, deltat = rms(d1), rms(d2), rms(dt)
            out = {
                "rms_h0": r0,
                "rms_h1": r1,
                "rms_h2": r2,
                "rms_h1_minus_h0": delta1,
                "rms_h2_minus_h1": delta2,
                "rms_h2_minus_h0": deltat,
                "R1": (delta1 / r0) if r0 else None,
                "R2": (delta2 / r1) if r1 else None,
                "Rtotal": (deltat / r0) if r0 else None,
                "finite_fraction_h2": float(torch.isfinite(h2).float().mean().item()),
                "finite_logits": None,
                "fc2_weight_norm": {
                    f"blocks.{i}": float(model.blocks[i].fc2.weight.detach().norm().item()) for i in (0, 1)
                },
                "fc2_bias_norm": {
                    f"blocks.{i}": float(model.blocks[i].fc2.bias.detach().norm().item()) for i in (0, 1)
                },
                "bond_proj_weight_norm": float(model.bond_proj.weight.detach().norm().item()),
                "cos_h0_h1": float(
                    torch.nn.functional.cosine_similarity(h0.flatten(), h1.flatten(), dim=0).item()
                ),
                "n_atoms": int(h0.shape[0]),
            }
            logits = model.post_local(h2, batch)
            out["finite_logits"] = bool(torch.isfinite(logits).all().item())
            out["logits_abs_max"] = float(logits.abs().max().item())
            out["logits_shape"] = list(logits.shape)
    finally:
        if was_training:
            model.train()
    return out


__all__ = [
    "MessageFFN",
    "W2DynamicModel",
    "block_fingerprints",
    "build_dynamic_model",
    "check_edges_intra_molecule",
    "compare_states",
    "initialize_dynamic_modules",
    "parameter_audit",
    "rms",
    "shuffle_src_within_molecule",
    "untrained_w2",
    "usage_metrics",
]
