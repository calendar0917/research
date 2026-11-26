"""FIXED1 with structural-unit message dropout (training only).

The inherited FIXED1 arithmetic is untouched::

    z1 = LN1(h0);  m1 = sum_j SiLU(z1[j] + e_ji);        h1 = h0 + FFN1(z1, m1)
    zt = LN2(h1);  zs = LN2(h0);  m2 = sum_j SiLU(zs[j] + e_ji);  h2 = h1 + FFN2(zt, m2)

``STRUCT_DROP`` only changes the aggregation of the directed message term::

    keep = (rand(n_physical_bonds, generator=mask_rng) >= q).float() / (1 - q)
    keep_directed = cat([keep, keep])          # exact directed_bonds() order
    p = SiLU(z_source[src] + edge_msg) * keep_directed[:, None]
    m = scatter_sum(p, dst)

* one Bernoulli draw per physical undirected bond and training batch;
* the same ``keep_directed`` tensor object is used by block 1 and block 2;
* eval mode never draws a mask and is bit-for-bit the inherited FIXED1 code
  path (delegated to ``DSModel.structural_layers``);
* no new parameter, no auxiliary loss, no prediction head.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

import ds_model as DM


def masked_source_message(
    z_source: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_msg: torch.Tensor,
    keep_directed: torch.Tensor | None,
    audit: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    """``m[i] = sum_j SiLU(z[j] + e_ji) * keep_ji`` over real directed bonds."""
    if src.numel() == 0:
        out = torch.zeros_like(z_source)
        if audit is not None:
            audit.append(
                {
                    "keep_id": None if keep_directed is None else id(keep_directed),
                    "keep_data_ptr": None if keep_directed is None else int(keep_directed.data_ptr()),
                    "keep_sum": None if keep_directed is None else float(keep_directed.sum().item()),
                    "keep_numel": None if keep_directed is None else int(keep_directed.numel()),
                    "n_directed": 0,
                }
            )
        return out
    p = F.silu(z_source.index_select(0, src) + edge_msg)
    if keep_directed is not None:
        if int(keep_directed.numel()) != int(src.numel()):
            raise RuntimeError(
                f"directed mask length {int(keep_directed.numel())} != directed edges {int(src.numel())}"
            )
        p = p * keep_directed.unsqueeze(1)
    if audit is not None:
        audit.append(
            {
                "keep_id": None if keep_directed is None else id(keep_directed),
                "keep_data_ptr": None if keep_directed is None else int(keep_directed.data_ptr()),
                "keep_sum": None if keep_directed is None else float(keep_directed.sum().item()),
                "keep_numel": None if keep_directed is None else int(keep_directed.numel()),
                "n_directed": int(src.numel()),
            }
        )
    return torch.zeros_like(z_source).index_add_(0, dst, p)


class StructuralDropModel(DM.DSModel):
    """``DSModel`` FIXED1 whose message term is dropout-masked while training."""

    def __init__(
        self,
        *args: Any,
        struct_drop_q: float = 0.10,
        mask_seed: int = 20261126,
        **kwargs: Any,
    ) -> None:
        kwargs["arm"] = "FIXED1"
        super().__init__(*args, **kwargs)
        if not (0.0 <= float(struct_drop_q) < 1.0):
            raise ValueError(f"invalid structural dropout q={struct_drop_q}")
        self.struct_drop_q = float(struct_drop_q)
        self.mask_seed = int(mask_seed)
        self._mask_gen = None
        self.mask_draws = 0
        self.eval_forward_calls = 0
        self.mask_events: list[dict[str, Any]] = []
        self.last_mask: dict[str, Any] | None = None

    # -- dedicated mask RNG ------------------------------------------------

    def mask_generator(self, device: torch.device) -> torch.Generator:
        if self._mask_gen is None or torch.device(self._mask_gen.device) != torch.device(device):
            gen = torch.Generator(device=torch.device(device))
            gen.manual_seed(int(self.mask_seed))
            self._mask_gen = gen
        return self._mask_gen

    def mask_rng_state(self) -> Any:
        if self._mask_gen is None:
            return None
        return self._mask_gen.get_state()

    def restore_mask_rng_state(self, state: Any) -> None:
        if state is None:
            return
        # the generator device follows the model device; deferred to first use
        if self._mask_gen is None:
            device = next(self.parameters()).device
            self._mask_gen = torch.Generator(device=device)
            self._mask_gen.manual_seed(int(self.mask_seed))
        self._mask_gen.set_state(state.to(self._mask_gen.device))

    # -- mask --------------------------------------------------------------

    def draw_keep(self, n_physical: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gen = self.mask_generator(device)
        u = torch.rand(int(n_physical), generator=gen, device=device)
        keep = (u >= float(self.struct_drop_q)).to(dtype) / (1.0 - float(self.struct_drop_q))
        self.mask_draws += 1
        return keep

    # -- arithmetic --------------------------------------------------------

    def structural_layers(
        self,
        h0: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        gamma: float | None = None,
        arm: str | None = None,
        c_override: torch.Tensor | None = None,
    ):
        if not self.training:
            # eval: no mask, no scaling, exactly the inherited FIXED1 code path
            self.eval_forward_calls += 1
            return super().structural_layers(
                h0, batch, edges=edges, gamma=gamma, arm=arm, c_override=c_override
            )
        name = self.arm if arm is None else str(arm)
        if name != "FIXED1":
            raise KeyError(f"structural dropout is defined for FIXED1 only, got {name}")
        src, dst, edge_msg = DM.directed_bonds(self, batch) if edges is None else edges
        n_directed = int(src.numel())
        if n_directed % 2:
            raise RuntimeError(f"directed edge count {n_directed} is not even")
        n_physical = n_directed // 2
        keep = self.draw_keep(n_physical, h0.device, h0.dtype)
        keep_directed = torch.cat([keep, keep], dim=0) if n_physical else keep
        audit: list[dict[str, Any]] = []
        blk1 = self.blocks[0]
        z1 = blk1.norm(h0)
        m1 = masked_source_message(z1, src, dst, edge_msg, keep_directed, audit)
        h1 = h0 + DM._apply_ffn(blk1, z1, m1)
        blk2 = self.blocks[1]
        z_target = blk2.norm(h1)
        z_source = blk2.norm(h0)
        m2 = masked_source_message(z_source, src, dst, edge_msg, keep_directed, audit)
        h2 = h1 + DM._apply_ffn(blk2, z_target, m2)
        event = {
            "n_physical_bonds": int(n_physical),
            "n_directed_edges": int(n_directed),
            "q": float(self.struct_drop_q),
            "kept_fraction_directed": float(keep_directed.mean().item()) if n_directed else 1.0,
            "dropped_fraction_physical": float((keep == 0).to(torch.float64).mean().item()) if n_physical else 0.0,
            "keep_sum": float(keep.sum().item()),
            "keep_numel": int(keep.numel()),
            "block_masks": audit,
            "same_mask_tensor_both_blocks": bool(
                len(audit) == 2 and audit[0]["keep_id"] is not None and audit[0]["keep_id"] == audit[1]["keep_id"]
            ),
            "directions_share_physical_mask": bool(
                n_physical > 0 and torch.equal(keep_directed[:n_physical], keep_directed[n_physical:])
            ),
        }
        self.last_mask = event
        self.mask_events.append({k: event[k] for k in ("n_physical_bonds", "kept_fraction_directed", "dropped_fraction_physical", "same_mask_tensor_both_blocks", "directions_share_physical_mask")})
        info: dict[str, Any] = {
            "arm": "FIXED1",
            "structural_dropout": True,
            "layer2_reads": "h0 of the real 1-hop neighbours + h1[i]",
            "m2": m2,
            "mask": event,
        }
        return h1, h2, info

    def finalize_checkpoint_extra(self) -> dict[str, Any]:
        return {
            "structural_dropout": {
                "q": float(self.struct_drop_q),
                "mask_seed": int(self.mask_seed),
                "mask_draws": int(self.mask_draws),
                "eval_forward_calls": int(self.eval_forward_calls),
                "scope": "both FIXED1 blocks, directed message term, shared per physical bond",
                "mask_rng_device": None if self._mask_gen is None else str(self._mask_gen.device),
            }
        }


__all__ = ["StructuralDropModel", "masked_source_message"]
