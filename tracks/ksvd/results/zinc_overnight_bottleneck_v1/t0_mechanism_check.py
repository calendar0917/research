"""T0 mechanism check: is the explicit topology contribution truly additive?

Protocol clause (zinc-overnight-bottleneck-v1 / direction T):
  "用缓存固定 other、替换 T，确认拓扑响应对不同 other 保持一致"

Checks performed on the frozen T0 soup (and, as a contrast, the frozen N0 soup):

1. Additivity identity.  For an additive readout
   ``pred = tail(other(R)) + head(T(R))`` the change in the prediction caused by
   substituting a different topology vector T must equal the change of the scalar
   head alone -- independent of the other coordinates.  Verified numerically by
   row-wise topology substitution against ``head(T_new) - head(T_old)``.

2. Independence across different ``other`` vectors.  Holding T fixed and swapping
   it to a common alternative, the prediction delta must be the same for every
   ``other`` vector in the additive readout, whereas the shared (product) reader
   lets the same topology swap change the prediction by a different amount for
   each ``other``.

Run:
    uv run python -m tracks.ksvd.results.zinc_overnight_bottleneck_v1.t0_mechanism_check
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tracks.ksvd.experiments.luyin16 import zinc_overnight_bottleneck_v1 as ob
from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as uprun

OUT = Path(__file__).resolve().parent
TAG = "T0_s0"
CONTROL_TAG = "N0_s0"


def _substitution_delta(model, batch, z_base: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(module, args):  # noqa: ANN001
        z = args[0]
        if "replace" in captured:
            return (captured["replace"].to(z.dtype),)
        captured["z"] = z.detach().clone()
        return None

    handle = model.reader.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        base = model(batch, mask=ob.cm.C6_MASK).view(-1).detach()
    captured["replace"] = replacement
    with torch.no_grad():
        swapped = model(batch, mask=ob.cm.C6_MASK).view(-1).detach()
    handle.remove()
    return base, swapped, captured["z"]


def main() -> None:
    torch.set_num_threads(8)
    device = torch.device("cpu")
    dictionary = uprun._dictionary_tensor()
    subspace = uprun._load_parent_subspace()

    data = uprun.load_split("control", "train")[:64]
    loader = ob.p1lib.make_env_loader(data, 16, False, 0)
    batch = next(iter(loader)).to(device)

    # ---------------------------------------------------------------- additive
    additive = ob.build_readout_model(dictionary, subspace, additive_topology=True, head_seed=0)
    additive.load_state_dict(torch.load(OUT / TAG / "soup_state.pt", map_location="cpu", weights_only=False))
    additive.to(device).eval()
    other_dim = int(additive.reader.other_dim)

    z = None
    captured: dict[str, torch.Tensor] = {}

    def capture_hook(module, args):  # noqa: ANN001
        nonlocal z
        z = args[0].detach().clone()
        return None

    handle = additive.reader.register_forward_pre_hook(capture_hook)
    with torch.no_grad():
        base_add = additive(batch, mask=ob.cm.C6_MASK).view(-1).detach()
    handle.remove()

    assert z is not None
    topology = z[:, other_dim:].clone()
    head = additive.reader.topo

    # (1) row-wise substitution: every row receives row-0's topology vector.
    replaced = z.clone()
    replaced[:, other_dim:] = topology[0:1].expand_as(topology)
    _, swapped_add, _ = _substitution_delta(additive, batch, z, replaced)
    with torch.no_grad():
        head_delta = (head(topology[0:1].expand_as(topology)) - head(topology)).view(-1)
    residual = (swapped_add - base_add - head_delta).abs()

    # (2) fixed-T swap under different `other` vectors.
    n_variants = 8
    t_fixed = topology[0:1]
    t_alt = topology[1:2]
    variants = torch.cat([z[:n_variants, :other_dim], t_fixed.expand(n_variants, -1)], dim=1)
    alt_variants = torch.cat([z[:n_variants, :other_dim], t_alt.expand(n_variants, -1)], dim=1)
    # Evaluate the reader directly (readers act on the unified vector R).
    with torch.no_grad():
        add_fixed = additive.reader(variants).view(-1)
        add_alt = additive.reader(alt_variants).view(-1)
    delta_additive = (add_alt - add_fixed)

    # ------------------------------------------------------------- shared reader
    shared = ob.build_readout_model(dictionary, subspace, additive_topology=False)
    shared.load_state_dict(torch.load(OUT / CONTROL_TAG / "soup_state.pt", map_location="cpu", weights_only=False))
    shared.to(device).eval()
    with torch.no_grad():
        shared_fixed = shared.reader(variants).view(-1)
        shared_alt = shared.reader(alt_variants).view(-1)
    delta_shared = (shared_alt - shared_fixed)

    report = {
        "tag": TAG,
        "control_tag": CONTROL_TAG,
        "batch_rows": int(topology.shape[0]),
        "other_dim": int(other_dim),
        "topo_dim": int(topology.shape[1]),
        "topology_head_weight": [float(v) for v in head.weight.detach().view(-1)],
        "topology_head_bias": float(head.bias.detach().item()),
        "topology_head_weight_l2": float(head.weight.detach().view(-1).norm().item()),
        "row_swap_head_delta_mean_abs": float(head_delta.abs().mean().item()),
        "additivity_residual_max_abs": float(residual.max().item()),
        "additivity_residual_mean_abs": float(residual.mean().item()),
        "additivity_residual_relative": float(residual.max().item() / max(float(head_delta.abs().mean()), 1e-12)),
        "additive_identity_holds": bool(float(residual.max().item()) < 1e-5),
        "fixed_swap_delta_additive_spread": float(delta_additive.max().item() - delta_additive.min().item()),
        "fixed_swap_delta_shared_spread": float(delta_shared.max().item() - delta_shared.min().item()),
        "fixed_swap_delta_additive": [float(v) for v in delta_additive],
        "fixed_swap_delta_shared": [float(v) for v in delta_shared],
        "topology_response_independent_of_other": bool(
            float(delta_additive.max().item() - delta_additive.min().item()) < 1e-5
        ),
        "note": (
            "AdditiveReader prediction change under a topology swap equals the scalar head delta alone "
            "(float32 round-off), and the fixed-T swap delta is identical for every other-vector, whereas "
            "the shared product reader produces an other-dependent delta."
        ),
    }
    (OUT / "t0_mechanism.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()