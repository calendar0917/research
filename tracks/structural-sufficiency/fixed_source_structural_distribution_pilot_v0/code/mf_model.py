"""MEANFIELD: FIXED1 with per-(molecule, bond-signature) mean-field messages.

The computation, per block ``l`` and per real directed edge ``j -> i``:

1. the FIXED1 message is computed **verbatim** (``p = SiLU(z_source[j] + e_ji)``);
2. the real directed edges are grouped by the joint key
   ``(graph_id, raw_bond_feature_signature)`` -- the *raw* 13-d bond feature
   tuple, never a learned embedding;
3. every edge's message content is replaced by its group mean
   ``group_mean = segment_sum(p, group_id) / group_count``, ``p_mf = group_mean[group_id]``;
4. ``m[i] = sum_{j -> i} p_mf`` (vectorised ``index_add_`` segment reduce);
5. ``h_next = h_current + FFN(z_target, m)`` -- the FIXED1 block arithmetic.

States read (identical to FIXED1; ``h1[src]`` / ``h2[src]`` are never read)::

    block 1:  z_source = LN1(h0),  z_target = LN1(h0)
    block 2:  z_source = LN2(h0),  z_target = LN2(h1)

No parameter, no re-computed local encoding, no random permutation is added.
What *is* given up is the per-edge correspondence between a distinct source
content and a distinct centre: all edges of one (molecule, bond-type) group
share one content vector, which is molecule-wide information sharing.

Directed edges of one physical bond share the same raw bond feature, so the two
directions of a bond always fall in the same group.  Consequently every group
size is even in directed-edge units (>= 2), the singleton ratio is 0, and the
mean-field content of a bond is direction-symmetric -- which is exactly why the
whole layer is invariant under a global direction flip (see ``mf_checks``).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import ds_common as C
import ds_model as DM

ARM = "MEANFIELD"


# ---------------------------------------------------------------------------
# grouping (vectorised, no per-molecule python loop)
# ---------------------------------------------------------------------------


def directed_bond_features(batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(raw_bond_feature, graph_id)`` for the directed edge order of ``DM.directed_bonds``.

    ``DM.directed_bonds`` builds ``src = cat([i0, i1])`` / ``dst = cat([i1, i0])``, so
    the two directions of physical bond ``b`` sit at positions ``b`` and ``B + b`` and
    both carry ``bond_feat[b]``.
    """
    bf = batch["bond_feat"]
    gb = batch["bond_batch"]
    return torch.cat([bf, bf], dim=0), torch.cat([gb, gb], dim=0)


def group_index(bf_dir: torch.Tensor, gb_dir: torch.Tensor) -> tuple[torch.Tensor, int, dict[str, Any]]:
    """Compact group id per directed edge for key ``(graph_id, bond_signature)``.

    The signature is the *full raw tuple*, obtained exactly with
    ``torch.unique(dim=0)``; there is no bucketing, hashing or learned embedding.
    """
    n_edges = int(bf_dir.shape[0])
    if n_edges == 0:
        empty = torch.zeros(0, dtype=torch.long, device=bf_dir.device)
        return empty, 0, {"n_edges": 0, "n_signatures": 0, "n_groups": 0}
    signature, signature_id = torch.unique(bf_dir, dim=0, return_inverse=True)
    n_sig = int(signature.shape[0])
    key = gb_dir.to(torch.long) * n_sig + signature_id.to(torch.long)
    _unique_key, group_id = torch.unique(key, return_inverse=True)
    n_groups = int(group_id.max().item()) + 1
    return group_id.to(torch.long), n_groups, {
        "n_edges": n_edges,
        "n_signatures": n_sig,
        "n_groups": int(n_groups),
    }


def group_sizes(group_id: torch.Tensor, n_groups: int) -> torch.Tensor:
    if n_groups == 0:
        return torch.zeros(0, dtype=torch.long, device=group_id.device)
    return torch.bincount(group_id, minlength=int(n_groups))


def meanfield_content(p: torch.Tensor, group_id: torch.Tensor, n_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(group_mean, p_meanfield)`` for one message matrix ``p`` of shape ``[E, W]``."""
    width = int(p.shape[1])
    if int(p.shape[0]) == 0:
        empty = p.new_zeros((0, width))
        return p.new_zeros((0, width)), empty
    summed = p.new_zeros((int(n_groups), width)).index_add_(0, group_id, p)
    counts = group_sizes(group_id, int(n_groups)).to(p.dtype)
    group_mean = summed / counts.clamp(min=1.0).unsqueeze(1)
    return group_mean, group_mean.index_select(0, group_id)


def meanfield_message(
    z_source: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_msg: torch.Tensor,
    group_id: torch.Tensor,
    n_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(m, p, p_meanfield)``; ``p`` is the untouched FIXED1 message content."""
    if int(src.numel()) == 0:
        width = int(z_source.shape[1])
        return (
            torch.zeros_like(z_source),
            z_source.new_zeros((0, width)),
            z_source.new_zeros((0, width)),
        )
    p = F.silu(z_source.index_select(0, src) + edge_msg)
    _group_mean, p_mf = meanfield_content(p, group_id, int(n_groups))
    m = torch.zeros_like(z_source).index_add_(0, dst, p_mf)
    return m, p, p_mf


# ---------------------------------------------------------------------------
# the computation
# ---------------------------------------------------------------------------


def meanfield_layers(
    model: nn.Module,
    h0: torch.Tensor,
    edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    group_id: torch.Tensor,
    n_groups: int,
    *,
    keep_messages: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """The MEANFIELD two-block computation on top of any FIXED1-shaped model."""
    src, dst, edge_msg = edges
    blk1, blk2 = model.blocks[0], model.blocks[1]

    z1 = blk1.norm(h0)
    m1, p1, p1_mf = meanfield_message(z1, src, dst, edge_msg, group_id, n_groups)
    h1 = h0 + DM._apply_ffn(blk1, z1, m1)

    z2_target = blk2.norm(h1)
    z2_source = blk2.norm(h0)
    m2, p2, p2_mf = meanfield_message(z2_source, src, dst, edge_msg, group_id, n_groups)
    h2 = h1 + DM._apply_ffn(blk2, z2_target, m2)

    info: dict[str, Any] = {
        "arm": ARM,
        "n_directed_edges": int(src.numel()),
        "n_groups": int(n_groups),
        "mean_group_size": float(src.numel() / n_groups) if n_groups else 0.0,
        "layer2_reads": "LN2(h0)[j] (group-mean field) + LN2(h1)[i]",
        "molecule_wide_information_sharing": True,
    }
    if keep_messages:
        info.update({"p1": p1, "p1_mf": p1_mf, "m1": m1, "p2": p2, "p2_mf": p2_mf, "m2": m2})
    else:
        info["m2"] = m2
    return h1, h2, info


class MFModel(DM.DSModel):
    """Frozen W2 + two FIXED1 blocks, with mean-field message content."""

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
        # ``DM.DSModel`` only uses ``arm`` for validation/recording; the module
        # set, the block wiring and every initialiser are arm-independent.
        super().__init__(spec, norm, n_out, cfg, arm="FIXED1", n_layers=n_layers, block_hidden=block_hidden)
        self.arm = ARM

    def structural_layers(
        self,
        h0: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        keep_messages: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        src, dst, edge_msg = DM.directed_bonds(self, batch) if edges is None else edges
        bf_dir, gb_dir = directed_bond_features(batch)
        group_id, n_groups, group_audit = group_index(bf_dir, gb_dir)
        h1, h2, info = meanfield_layers(
            self, h0, (src, dst, edge_msg), group_id, n_groups, keep_messages=keep_messages
        )
        info.update(group_audit)
        return h1, h2, info

    # FIXED1 diagnostics call ``stage1``/``stage2``; keep them consistent with
    # the mean-field arithmetic instead of inheriting the FIXED1 bodies.
    def stage1(self, h0, src, dst, edge_msg, *, arm: str | None = None, **_: Any):
        z1 = self.blocks[0].norm(h0)
        raise NotImplementedError("use structural_layers(); MEANFIELD groups once per block")


# ---------------------------------------------------------------------------
# frozen-weight intervention (no training, no weight change)
# ---------------------------------------------------------------------------


def fixed1_messages(
    model: nn.Module,
    h0: torch.Tensor,
    edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FIXED1 ``(m1, m2, p1, p2)`` for the same weights, on the FIXED1 states."""
    src, dst, edge_msg = edges
    blk1, blk2 = model.blocks[0], model.blocks[1]
    z1 = blk1.norm(h0)
    m1 = torch.zeros_like(z1).index_add_(0, dst, F.silu(z1.index_select(0, src) + edge_msg))
    h1 = h0 + DM._apply_ffn(blk1, z1, m1)
    z2_source = blk2.norm(h0)
    z2_target = blk2.norm(h1)
    p2 = F.silu(z2_source.index_select(0, src) + edge_msg)
    m2 = torch.zeros_like(z2_source).index_add_(0, dst, p2)
    h2 = h1 + DM._apply_ffn(blk2, z2_target, m2)
    return m1, m2, p2, h2


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def build_mf_model(
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    verify: bool = True,
) -> tuple[MFModel, dict[str, Any]]:
    """Build MEANFIELD from the *original FIXED1* constructor state of ``seed``.

    Identical rule to ``ds_model.build_ds_model``: the W2 part comes from
    ``bs_model.build_model`` (never re-derived), the new modules are constructed
    inside a forked RNG and filled from the historical private generator, with
    both ``fc2`` weights/biases zeroed so epoch 0 reproduces untrained W2.
    """
    cfg = dict(C.BC.config_for_scale(2))
    reference, ref_audit = C.BM.build_model(
        "W2", spec, norm, int(n_out), seed=int(seed), device="cpu", verify_parameters=True
    )
    rng_after_w2 = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = MFModel(spec, norm, int(n_out), cfg)
    if not torch.equal(rng_after_w2, torch.get_rng_state()):
        raise RuntimeError("MFModel construction advanced the global torch RNG")
    load = model.load_state_dict(reference.state_dict(), strict=False)
    expected_missing = {k for k in model.state_dict() if C.is_new_parameter(k)}
    if set(load.missing_keys) != expected_missing or load.unexpected_keys:
        raise RuntimeError(f"W2 state load mismatch: {sorted(load.missing_keys)}")
    base_before = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    init_audit = C.initialize_dynamic_modules(model, seed=int(seed), zero_residual=True)
    base_after = C.state_fingerprint(C.trunk_subset(model.state_dict()))
    if base_before != base_after:
        raise RuntimeError("new-module initialisation changed the W2 part of the state")
    model.to(torch.device(device))
    audit = C.parameter_audit(model)
    if verify and (not audit["all_match"] or int(audit["total"]) != 1_128_468):
        raise RuntimeError(f"parameter audit failed for {ARM}: {audit['checks']}")
    ref_state = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
    ours_base = C.state_subset({k: v.detach().cpu() for k, v in model.state_dict().items()}, new=False)
    base_equal = C.compare_states(ours_base, ref_state)
    if verify and not base_equal["all_equal"]:
        raise RuntimeError("W2 part is not bitwise identical to the W2 constructor")
    out = {
        "arm": ARM,
        "seed": int(seed),
        "widths": dict(cfg),
        "parameters": audit,
        "w2_part_bitwise_equal_reference": base_equal,
        "w2_trunk_init_fingerprint": base_before,
        "base_state_sha256": base_before,
        "new_state_sha256": C.state_fingerprint(C.state_subset(model.state_dict(), new=True)),
        "init": init_audit,
        "n_state_tensors": len(model.state_dict()),
        "device": str(device),
        "information_access_note": (
            "mean-field content per (molecule, bond signature) group; molecule-wide "
            "information sharing, not information-access-equivalent to FIXED1"
        ),
    }
    if verify and int(seed) == 0:
        recorded = C.W2_SEED0["trunk_init_fingerprint"]
        if base_before != recorded:
            raise RuntimeError(f"W2 trunk fingerprint {base_before} != recorded W2 seed-0 {recorded}")
    return model, out


__all__ = [
    "ARM",
    "MFModel",
    "build_mf_model",
    "directed_bond_features",
    "fixed1_messages",
    "group_index",
    "group_sizes",
    "meanfield_content",
    "meanfield_layers",
    "meanfield_message",
]
