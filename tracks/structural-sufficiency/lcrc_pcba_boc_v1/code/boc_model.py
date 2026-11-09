"""LCRC-BOC: bond-centred objects and ring systems in the atom main state.

The W2 trunk is constructed exactly as in the historical seed-0 run and is
never fine-tuned from a trained checkpoint.  One atom encode, one bond encode
and one ``local`` pass produce ``a``, ``b`` and ``h0``.  The new object
encoders read only ``x = SiLU(Linear(192,64)(a))``, ``y = SiLU(Linear(192,64)(b))``
and the static structural cache.  They never read ``h0`` or the fused ``h`` and
nothing is propagated bond -> atom -> bond.

Main path::

    concat(h0_i [128], m_i [64], r_i [32]) -> Linear(224,256) -> SiLU
                                           -> Linear(256,128) -> LayerNorm
    h_i = W2.dropout(h_i)
    logits = W2.post_local(h_i, batch)

``FULL`` uses the real ring-system vector; ``BOND`` passes exact zeros.  Both
arms instantiate the same modules with the same shapes and the same private
initialisation; no output layer is zeroed.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import boc_common as C


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seg_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = values.new_zeros((int(size), values.shape[1]))
    if values.numel():
        out.index_add_(0, index, values)
    return out


def _sqrt_clamp(count: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(count.clamp(min=1).to(torch.float32)).unsqueeze(1)


def _log1p_col(count: torch.Tensor) -> torch.Tensor:
    return torch.log1p(count.clamp(min=0).to(torch.float32)).unsqueeze(1)


def _std_linear_init(linear: nn.Linear, generator: torch.Generator) -> None:
    nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    if linear.bias is not None:
        bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
        linear.bias.uniform_(-bound, bound, generator=generator)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class BOCModel(C.BSM.BSModel):
    """W2 trunk + one-shot bond-object / ring-system main-state composition."""

    def __init__(self, spec, norm, n_out: int, cfg: Mapping[str, Any], *, arm: str):
        super().__init__(spec, norm, n_out, cfg)
        if arm not in C.MODELS:
            raise KeyError(arm)
        self.arm = str(arm)
        self.width = int(self.cfg["width"])
        self.atom_out = int(self.cfg["atom_out"])
        self.obj = int(C.OBJ)

        # shared initial projections (the same x/y are used by every object)
        self.boc_atom_proj = nn.Linear(self.atom_out, self.obj)
        self.boc_bond_proj = nn.Linear(self.atom_out, self.obj)

        # bond-centred object
        self.q_mlp = C.TU.MLP([3 * self.obj, 96, self.obj])
        self.t_mlp = C.TU.MLP([3 * self.obj + 3 * self.obj + 1, 96, self.obj])
        self.same_mlp = C.TU.MLP([3 * self.obj + self.obj + 1, 96, self.obj])
        self.cross_mlp = C.TU.MLP([3 * self.obj + self.obj + 1, 96, self.obj])
        self.bond_mlp = C.TU.MLP([6 * self.obj + 5, 128, self.obj])
        self.bond_norm = nn.LayerNorm(self.obj)

        # ring-system object
        self.ring_ext_proj = nn.Linear(2 * self.obj, self.obj)
        self.ring_atom_proj = nn.Linear(2 * self.obj, self.obj)
        self.ring_dist = nn.Embedding(int(C.N_RING_BUCKETS), int(C.RING_DIST_EMB))
        self.ring_pair_mlp = C.TU.MLP([3 * self.obj + self.obj + int(C.RING_DIST_EMB) + 1, 96, int(C.RING_PAIR)])
        self.ring_out = C.TU.MLP([int(C.RING_PAIR) + self.obj + 3, 64, int(C.RING_PAIR)])
        self.ring_norm = nn.LayerNorm(int(C.RING_PAIR))

        # main fusion: no zeroed output layer
        self.fusion = nn.Sequential(
            nn.Linear(self.width + self.obj + int(C.RING_PAIR), 256),
            nn.SiLU(),
            nn.Linear(256, self.width),
        )
        self.fusion_norm = nn.LayerNorm(self.width)

    # -- object encoders -------------------------------------------------

    def _bond_objects(self, batch: Mapping[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):
        idx = batch["bond_index"]
        n_bonds = int(idx.shape[0])
        if n_bonds == 0:
            zero = x.new_zeros((0, self.obj))
            return zero, {"z_b": zero}
        u = idx[:, 0].to(torch.long)
        v = idx[:, 1].to(torch.long)
        y_pad = torch.cat([y, y.new_zeros((1, self.obj))], dim=0)
        nb_ptr = batch["boc_nb_ptr"].to(torch.long)
        pair_ptr = batch["boc_pair_ptr"].to(torch.long)
        nb_counts = nb_ptr[1:] - nb_ptr[:-1]
        pair_counts = pair_ptr[1:] - pair_ptr[:-1]
        nb_center = torch.repeat_interleave(torch.arange(n_bonds, device=x.device), nb_counts)
        pair_center = torch.repeat_interleave(torch.arange(n_bonds, device=x.device), pair_counts)

        # 4.1 centre anchor
        q_forward = self.q_mlp(torch.cat([x[u], x[v], y], dim=1))
        q_reverse = self.q_mlp(torch.cat([x[v], x[u], y], dim=1))

        # 4.2 two-sided neighbours
        side = batch["boc_nb_side"].to(torch.long)
        nb_atom = batch["boc_nb_atom"].to(torch.long)
        nb_bond_side = batch["boc_nb_bond_side"].to(torch.long)
        nb_bond_other = batch["boc_nb_bond_other"].to(torch.long)
        cu = torch.where(side == 0, u[nb_center], v[nb_center])
        cv = torch.where(side == 0, v[nb_center], u[nb_center])
        has_other = (nb_bond_other >= 0).to(torch.float32).unsqueeze(1)
        other_pad = torch.where(
            nb_bond_other >= 0, nb_bond_other, torch.full_like(nb_bond_other, y.shape[0])
        )
        t = self.t_mlp(
            torch.cat(
                [
                    x[cu],
                    x[cv],
                    x[nb_atom],
                    y[nb_center],
                    y[nb_bond_side],
                    y_pad[other_pad],
                    has_other,
                ],
                dim=1,
            )
        )
        side_l = (side == 0).to(torch.float32).unsqueeze(1)
        side_r = 1.0 - side_l
        s_l = _seg_sum(t * side_l, nb_center, n_bonds) / _sqrt_clamp(batch["boc_nL"].float())
        s_r = _seg_sum(t * side_r, nb_center, n_bonds) / _sqrt_clamp(batch["boc_nR"].float())

        # 4.3 same-side forks (one encoder for LL and RR)
        kind = batch["boc_pair_kind"].to(torch.long)
        pair_a = batch["boc_pair_a"].to(torch.long)
        pair_b = batch["boc_pair_b"].to(torch.long)
        pair_bond = batch["boc_pair_bond"].to(torch.long)
        t1 = t[pair_a]
        t2 = t[pair_b]
        pair_is_bond = (pair_bond >= 0).to(torch.float32).unsqueeze(1)
        pair_bond_pad = torch.where(
            pair_bond >= 0, pair_bond, torch.full_like(pair_bond, y.shape[0])
        )
        y_pair = y_pad[pair_bond_pad]

        same_mask = kind <= 1
        if bool(same_mask.any()):
            a_s = t1[same_mask]
            b_s = t2[same_mask]
            p_same = self.same_mlp(
                torch.cat(
                    [
                        a_s + b_s,
                        (a_s - b_s).abs(),
                        a_s * b_s,
                        y_pair[same_mask],
                        pair_is_bond[same_mask],
                    ],
                    dim=1,
                )
            )
            center_same = pair_center[same_mask]
            kind_same = kind[same_mask]
            c_ll = _seg_sum(p_same[kind_same == 0], center_same[kind_same == 0], n_bonds)
            c_rr = _seg_sum(p_same[kind_same == 1], center_same[kind_same == 1], n_bonds)
        else:
            c_ll = x.new_zeros((n_bonds, self.obj))
            c_rr = x.new_zeros((n_bonds, self.obj))
        c_ll = c_ll / _sqrt_clamp(batch["boc_nLL"].float())
        c_rr = c_rr / _sqrt_clamp(batch["boc_nRR"].float())

        # 4.4 cross-side branches; the reverse direction is a real second MLP
        # input (the encoder is not symmetric) and is averaged into z_b.
        cross_mask = kind == 2
        if bool(cross_mask.any()):
            a_c = t1[cross_mask]
            b_c = t2[cross_mask]
            y_c = y_pair[cross_mask]
            e_c = pair_is_bond[cross_mask]
            inp_f = torch.cat([a_c, b_c, a_c * b_c, y_c, e_c], dim=1)
            inp_r = torch.cat([b_c, a_c, b_c * a_c, y_c, e_c], dim=1)
            p_cross = self.cross_mlp(torch.cat([inp_f, inp_r], dim=0))
            n_cross = int(a_c.shape[0])
            center_c = pair_center[cross_mask]
            c_lr = _seg_sum(p_cross[:n_cross], center_c, n_bonds)
            c_lr_rev = _seg_sum(p_cross[n_cross:], center_c, n_bonds)
        else:
            c_lr = x.new_zeros((n_bonds, self.obj))
            c_lr_rev = x.new_zeros((n_bonds, self.obj))
        c_lr = c_lr / _sqrt_clamp(batch["boc_nLR"].float())
        c_lr_rev = c_lr_rev / _sqrt_clamp(batch["boc_nLR"].float())

        # 4.5 final bond object, averaged over the two endpoint orders
        def features(q, s_left, s_right, c_left, c_right, c_cross, n_left, n_right, n_ll, n_rr, n_lr):
            return torch.cat(
                [
                    q,
                    s_left,
                    s_right,
                    c_left,
                    c_right,
                    c_cross,
                    _log1p_col(n_left),
                    _log1p_col(n_right),
                    _log1p_col(n_ll),
                    _log1p_col(n_rr),
                    _log1p_col(n_lr),
                ],
                dim=1,
            )

        n_l = batch["boc_nL"]
        n_r = batch["boc_nR"]
        n_ll = batch["boc_nLL"]
        n_rr = batch["boc_nRR"]
        n_lr = batch["boc_nLR"]
        f_forward = features(q_forward, s_l, s_r, c_ll, c_rr, c_lr, n_l, n_r, n_ll, n_rr, n_lr)
        f_reverse = features(q_reverse, s_r, s_l, c_rr, c_ll, c_lr_rev, n_r, n_l, n_rr, n_ll, n_lr)
        if f_forward.shape[1] != 6 * self.obj + 5:
            raise RuntimeError(f"bond feature width {f_forward.shape[1]} != {6 * self.obj + 5}")
        both = self.bond_mlp(torch.cat([f_forward, f_reverse], dim=0))
        both = self.bond_norm(both)
        z_b = 0.5 * (both[:n_bonds] + both[n_bonds:])
        return z_b, {"z_b": z_b, "t": t}

    def _ring_system(self, batch: Mapping[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor):
        n_atoms = int(x.shape[0])
        n_rings = int(batch["boc_n_rings"])
        if n_rings == 0:
            return x.new_zeros((n_atoms, int(C.RING_PAIR))), {"r": x.new_zeros((n_atoms, int(C.RING_PAIR)))}
        ra = batch["boc_ring_atom"].to(torch.long)
        y_pad = torch.cat([y, y.new_zeros((1, self.obj))], dim=0)
        ext_ptr = batch["boc_ring_ext_ptr"].to(torch.long)
        ext_counts = ext_ptr[1:] - ext_ptr[:-1]
        ext_owner = torch.repeat_interleave(torch.arange(int(ra.numel()), device=x.device), ext_counts)
        ext_atom = batch["boc_ring_ext_atom"].to(torch.long)
        ext_bond = batch["boc_ring_ext_bond"].to(torch.long)
        # 5.2 external substituent environment
        e = F.silu(self.ring_ext_proj(torch.cat([x[ext_atom], y[ext_bond]], dim=1)))
        u_ext = _seg_sum(e, ext_owner, int(ra.numel())) / _sqrt_clamp(ext_counts.float())
        t_i = F.silu(self.ring_atom_proj(torch.cat([x[ra], u_ext], dim=1)))

        # 5.3 intra-system atom-pair relation
        pi = batch["boc_ring_pair_i"].to(torch.long)
        pj = batch["boc_ring_pair_j"].to(torch.long)
        pd = batch["boc_ring_pair_d"].to(torch.long)
        pb = batch["boc_ring_pair_bond"].to(torch.long)
        ti = t_i[pi]
        tj = t_i[pj]
        pb_pad = torch.where(pb >= 0, pb, torch.full_like(pb, y.shape[0]))
        p_ring = self.ring_pair_mlp(
            torch.cat(
                [
                    ti + tj,
                    (ti - tj).abs(),
                    ti * tj,
                    y_pad[pb_pad],
                    self.ring_dist(pd),
                    (pb >= 0).to(torch.float32).unsqueeze(1),
                ],
                dim=1,
            )
        )
        ring_atom_system = batch["boc_ring_atom_system"].to(torch.long)
        ring_pair_system = batch["boc_ring_pair_system"].to(torch.long)
        pair_sum = _seg_sum(p_ring, ring_pair_system, n_rings)
        node_sum = _seg_sum(t_i, ring_atom_system, n_rings)
        n_atoms_sys = batch["boc_ring_atom_counts"].float()
        n_pairs_sys = (batch["boc_ring_pair_ptr"][1:] - batch["boc_ring_pair_ptr"][:-1]).float()
        mean_pair = pair_sum / n_pairs_sys.clamp(min=1).unsqueeze(1)
        mean_node = node_sum / n_atoms_sys.clamp(min=1).unsqueeze(1)
        scale = torch.cat(
            [
                _log1p_col(n_atoms_sys),
                _log1p_col(batch["boc_ring_n_edges"].float()),
                _log1p_col(batch["boc_ring_n_ext"].float()),
            ],
            dim=1,
        )
        r_c = self.ring_norm(self.ring_out(torch.cat([mean_pair, mean_node, scale], dim=1)))
        r = x.new_zeros((n_atoms, int(C.RING_PAIR)))
        r = r.index_add(0, ra, r_c[ring_atom_system])
        return r, {"ring_atom": ra, "ring_pair": p_ring, "r_c": r_c}

    # -- forward ---------------------------------------------------------

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_aux: bool = False,
        ring_zero: bool = False,
        bond_zero: bool = False,
    ):
        a, b, scal = self._encode(batch)
        h0 = self.local(batch, a, b, scal)  # exactly one W2 local pass, no dropout
        n_atoms = int(a.shape[0])
        x = F.silu(self.boc_atom_proj(a))
        y = F.silu(self.boc_bond_proj(b))
        z_b, bond_aux = self._bond_objects(batch, x, y)
        if bond_zero:
            z_b = torch.zeros_like(z_b)

        idx = batch["bond_index"]
        if int(idx.shape[0]):
            u = idx[:, 0].to(torch.long)
            v = idx[:, 1].to(torch.long)
            deg = a.new_zeros(n_atoms)
            deg = deg.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
            deg = deg.index_add_(0, v, torch.ones_like(u, dtype=a.dtype))
            m_sum = a.new_zeros((n_atoms, self.obj)).index_add_(0, u, z_b).index_add_(0, v, z_b)
            m = m_sum / _sqrt_clamp(deg)
        else:
            m = a.new_zeros((n_atoms, self.obj))
        if bond_zero:
            m = torch.zeros_like(m)

        use_ring = self.arm == "FULL" and not bool(ring_zero)
        if use_ring:
            r, ring_aux = self._ring_system(batch, x, y)
        else:
            r = a.new_zeros((n_atoms, int(C.RING_PAIR)))
            ring_aux = {}
        fusion_input = torch.cat([h0, m, r], dim=1)
        if fusion_input.shape[1] != self.width + self.obj + int(C.RING_PAIR):
            raise RuntimeError(f"fusion input width {fusion_input.shape[1]}")
        h = self.fusion_norm(self.fusion(fusion_input))
        h = self.drop(h)  # the original W2 dropout, exactly once
        logits = self.post_local(h, batch)
        if return_aux:
            aux = {
                "h0": h0,
                "h": h,
                "m": m,
                "r": r,
                "z_b": z_b,
                "fusion_input": fusion_input,
                "ring_zero": bool(not use_ring),
                "ring_nonzero": int(torch.count_nonzero(r).item()),
            }
            aux.update({f"bond_{k}": v for k, v in bond_aux.items() if k != "z_b"})
            aux.update({f"ring_{k}": v for k, v in ring_aux.items()})
            return logits, aux
        return logits

    # -- diagnostics -----------------------------------------------------

    @torch.no_grad()
    def channel_norms(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        try:
            logits, aux = self(batch, return_aux=True)
        finally:
            self.train(was_training)
        out: dict[str, Any] = {}
        for name in ("h0", "m", "r", "z_b", "fusion_input", "h"):
            v = aux[name].detach().float()
            out[name] = {
                "rms": float(v.pow(2).mean().sqrt().item()) if v.numel() else 0.0,
                "mean_abs": float(v.abs().mean().item()) if v.numel() else 0.0,
                "max_abs": float(v.abs().max().item()) if v.numel() else 0.0,
                "numel": int(v.numel()),
            }
        out["logits_rms"] = float(logits.detach().float().pow(2).mean().sqrt().item())
        out["ring_nonzero"] = int(aux["ring_nonzero"])
        return out


# ---------------------------------------------------------------------------
# initialisation: historical W2 exactly, private generator for new tensors
# ---------------------------------------------------------------------------


def new_keys(model: nn.Module, base_keys: set[str]) -> list[str]:
    return sorted(k for k in model.state_dict() if k not in base_keys)


def initialize_new_modules(model: BOCModel, *, seed: int) -> dict[str, Any]:
    """Initialise every new tensor with one private CPU generator.

    The global torch RNG is checked bitwise before/after.  No output layer is
    zeroed (the fusion output and the bond/ring outputs are ordinary Linear
    initialisations).
    """
    before = torch.get_rng_state().clone()
    rng_before = C.rng_state_hashes(C.capture_rng_state())
    gen = torch.Generator(device="cpu").manual_seed(
        int(seed) * 1_000_003 + int(C.MODELS[model.arm]["private_generator_offset"])
    )
    with torch.no_grad():
        for module in (
            model.boc_atom_proj,
            model.boc_bond_proj,
            model.ring_ext_proj,
            model.ring_atom_proj,
            model.bond_norm,
            model.ring_norm,
            model.fusion_norm,
        ):
            if isinstance(module, nn.Linear):
                _std_linear_init(module, gen)
        for mlp in (
            model.q_mlp,
            model.t_mlp,
            model.same_mlp,
            model.cross_mlp,
            model.bond_mlp,
            model.ring_pair_mlp,
            model.ring_out,
        ):
            for linear in mlp.lins:
                _std_linear_init(linear, gen)
        for linear in model.fusion:
            if isinstance(linear, nn.Linear):
                _std_linear_init(linear, gen)
        model.ring_dist.weight.normal_(0.0, 1.0, generator=gen)
    if not torch.equal(before, torch.get_rng_state()):
        raise RuntimeError("new-module initialisation changed the global torch RNG state")
    if C.rng_state_hashes(C.capture_rng_state()) != rng_before:
        raise RuntimeError("new-module initialisation changed a tracked RNG state")
    fusion_last = model.fusion[-1]
    if not isinstance(fusion_last, nn.Linear):
        raise RuntimeError("fusion output is not a Linear layer")
    return {
        "kind": "private_generator_standard_init",
        "private_generator_seed": int(seed) * 1_000_003 + int(C.MODELS[model.arm]["private_generator_offset"]),
        "linear_rule": "kaiming_uniform(a=sqrt(5)) + fan_in uniform bias",
        "ring_distance_embedding": "normal(0,1)",
        "layer_norm": "default (weight=1, bias=0)",
        "fusion_output_zeroed": False,
        "fusion_output_nonzero": int(torch.count_nonzero(fusion_last.weight).item()) > 0
        or int(torch.count_nonzero(fusion_last.bias).item()) > 0,
        "global_rng_unchanged": True,
    }


def parameter_inventory(model: nn.Module) -> dict[str, Any]:
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        top = name.split(".", 1)[0]
        if top in (
            "boc_atom_proj",
            "boc_bond_proj",
            "q_mlp",
            "t_mlp",
            "same_mlp",
            "cross_mlp",
            "bond_mlp",
            "bond_norm",
        ):
            key = "bond_channel"
        elif top in (
            "ring_ext_proj",
            "ring_atom_proj",
            "ring_dist",
            "ring_pair_mlp",
            "ring_out",
            "ring_norm",
        ):
            key = "ring_channel"
        elif top in ("fusion", "fusion_norm"):
            key = "main_fusion"
        else:
            key = "w2_trunk"
        groups[key] = groups.get(key, 0) + int(p.numel())
    groups["total"] = int(sum(groups.values()))
    return groups


def build_model(
    arm: str,
    spec,
    norm: Mapping[str, Any],
    n_out: int = C.N_TASKS,
    *,
    seed: int = C.SEED,
    device: torch.device | str = "cpu",
    verify_parameters: bool = True,
) -> tuple[BOCModel, dict[str, Any]]:
    """Build one BOC arm on top of the frozen untrained W2 seed-0 state."""
    if arm not in C.MODELS:
        raise KeyError(arm)
    base_model, base_audit = C.BSM.build_model(
        C.BASE_MODEL_KEY, spec, norm, int(n_out), seed=int(seed), device="cpu"
    )
    cfg = dict(base_model.cfg)
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    base_keys = set(base_state)
    global_rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = BOCModel(spec, norm, int(n_out), cfg, arm=arm)
    added = new_keys(model, base_keys)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if set(missing) != set(added) or unexpected:
        raise RuntimeError(
            f"{arm}: base state load mismatch: missing={sorted(missing)} "
            f"expected_new={sorted(added)} unexpected={sorted(unexpected)}"
        )
    init_audit = initialize_new_modules(model, seed=int(seed))

    subset_diff: dict[str, float] = {}
    state = model.state_dict()
    for key in sorted(base_keys):
        if tuple(state[key].shape) != tuple(base_state[key].shape) or state[key].dtype != base_state[key].dtype:
            raise RuntimeError(f"{arm}: original W2 tensor shape/dtype changed: {key}")
        if not torch.equal(state[key].cpu(), base_state[key].cpu()):
            subset_diff[key] = float((state[key].float().cpu() - base_state[key].float().cpu()).abs().max().item())
    if subset_diff:
        raise RuntimeError(f"{arm}: original W2 parameters are not bitwise equal: {subset_diff}")
    if not torch.equal(global_rng_before, torch.get_rng_state()):
        raise RuntimeError(f"{arm}: build advanced the global torch RNG state")

    model.to(torch.device(device))
    base_elements = int(sum(p.numel() for n, p in model.named_parameters() if n in base_keys))
    new_elements = int(sum(p.numel() for n, p in model.named_parameters() if n not in base_keys))
    total = int(C.parameter_count(model))
    inventory = parameter_inventory(model)
    audit: dict[str, Any] = {
        "model": arm,
        "model_name": C.MODELS[arm]["name"],
        "role": C.MODELS[arm]["role"],
        "ring_input": C.MODELS[arm]["ring_input"],
        "seed": int(seed),
        "base_model": C.BASE_MODEL_KEY,
        "base_parameters": base_elements,
        "new_parameters": new_elements,
        "total_parameters": total,
        "expected_base_parameters": int(C.BASE_PARAMETERS),
        "expected_total_parameters": int(C.EXPECTED_PARAMETERS.get(arm, -1)),
        "parameter_inventory": inventory,
        "parameter_band": list(C.PARAMETER_BAND),
        "widths": cfg,
        "object_widths": {
            "x_projection": [int(C.ATOM_WIDTH), int(C.OBJ)],
            "y_projection": [int(C.ATOM_WIDTH), int(C.OBJ)],
            "q_mlp": [3 * int(C.OBJ), 96, int(C.OBJ)],
            "t_mlp": [3 * int(C.OBJ) + 3 * int(C.OBJ) + 1, 96, int(C.OBJ)],
            "same_mlp": [3 * int(C.OBJ) + int(C.OBJ) + 1, 96, int(C.OBJ)],
            "cross_mlp": [3 * int(C.OBJ) + int(C.OBJ) + 1, 96, int(C.OBJ)],
            "bond_mlp": [6 * int(C.OBJ) + 5, 128, int(C.OBJ)],
            "ring_pair_mlp": [
                3 * int(C.OBJ) + int(C.OBJ) + int(C.RING_DIST_EMB) + 1,
                96,
                int(C.RING_PAIR),
            ],
            "ring_out": [int(C.RING_PAIR) + int(C.OBJ) + 3, 64, int(C.RING_PAIR)],
            "fusion": [int(C.HIDDEN_WIDTH) + int(C.OBJ) + int(C.RING_PAIR), 256, int(C.HIDDEN_WIDTH)],
        },
        "init_rule": (
            "W2: historical seed-0 C1 state exactly; new modules: private CPU "
            "generator standard Linear init; no zeroed output layer"
        ),
        "new_module_init": init_audit,
        "new_state_keys": added,
        "new_parameter_inventory": {
            k: {"shape": list(state[k].shape), "numel": int(state[k].numel())} for k in added
        },
        "original_w2_subset": {"n_tensors": len(base_keys), "all_bitwise_equal": True, "n_differing": 0},
        "base_trunk_init_fingerprint": base_audit.get("trunk_init_fingerprint"),
        "base_full_trunk_init_fingerprint": base_audit.get("full_trunk_init_fingerprint"),
        "base_state_sha256": C.state_sha256(base_state),
        "model_state_sha256": C.state_sha256(state),
        "state_keys": sorted(state),
        "n_state_tensors": len(state),
        "parameter_tensors": len(list(model.parameters())),
        "global_rng_unchanged": True,
        "base_model_audit": base_audit,
    }
    if verify_parameters:
        if base_elements != int(C.BASE_PARAMETERS):
            raise RuntimeError(f"{arm}: base parameters {base_elements} != {C.BASE_PARAMETERS}")
        if total > int(C.PARAMETER_BAND[1]):
            raise RuntimeError(f"{arm}: {total} parameters exceed the frozen ceiling {C.PARAMETER_BAND[1]}")
        expected = C.EXPECTED_PARAMETERS.get(arm)
        if expected is not None and total != int(expected):
            raise RuntimeError(f"{arm}: total parameters {total} != recorded {expected}")
        if audit["base_state_sha256"] != C.W2["base_state_sha256_at_init"]:
            raise RuntimeError(
                f"{arm}: W2 init state {audit['base_state_sha256']} != historical "
                f"{C.W2['base_state_sha256_at_init']}"
            )
        if audit["base_trunk_init_fingerprint"] != C.W2["trunk_init_fingerprint"]:
            raise RuntimeError(f"{arm}: trunk fingerprint is not the historical seed-0 one")
        audit["parameter_check"] = "exact"
        audit["historical_init_check"] = "exact"
    C.EXPECTED_PARAMETERS.setdefault(arm, total)
    return model, audit


# ---------------------------------------------------------------------------
# call counting / gradient diagnostics
# ---------------------------------------------------------------------------


def count_module_calls(model: BOCModel, batch: Mapping[str, torch.Tensor]) -> dict[str, int]:
    names = (
        "atom_encoder",
        "bond_encoder",
        "local",
        "drop",
        "f",
        "rho",
        "readout",
        "boc_atom_proj",
        "boc_bond_proj",
        "q_mlp",
        "t_mlp",
        "same_mlp",
        "cross_mlp",
        "bond_mlp",
        "ring_ext_proj",
        "ring_atom_proj",
        "ring_pair_mlp",
        "ring_out",
        "fusion",
    )
    counters = {n: 0 for n in names}
    handles = []
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue

        def _hook(_m, _inp, _out, key=name):
            counters[key] += 1

        handles.append(module.register_forward_hook(_hook))
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(batch)
    finally:
        model.train(was_training)
        for h in handles:
            h.remove()
    return counters


def gradient_report(model: BOCModel) -> dict[str, Any]:
    groups = {
        "bond_channel": 0.0,
        "ring_channel": 0.0,
        "main_fusion": 0.0,
        "w2_trunk": 0.0,
    }
    nonzero = {k: 0 for k in groups}
    counts = {k: 0 for k in groups}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        top = name.split(".", 1)[0]
        if top in (
            "boc_atom_proj",
            "boc_bond_proj",
            "q_mlp",
            "t_mlp",
            "same_mlp",
            "cross_mlp",
            "bond_mlp",
            "bond_norm",
        ):
            key = "bond_channel"
        elif top in (
            "ring_ext_proj",
            "ring_atom_proj",
            "ring_dist",
            "ring_pair_mlp",
            "ring_out",
            "ring_norm",
        ):
            key = "ring_channel"
        elif top in ("fusion", "fusion_norm"):
            key = "main_fusion"
        else:
            key = "w2_trunk"
        g = p.grad.detach().float()
        groups[key] += float(g.pow(2).sum().item())
        nonzero[key] += int(torch.count_nonzero(g).item())
        counts[key] += int(g.numel())
    return {
        "grad_l2": {k: math.sqrt(v) for k, v in groups.items()},
        "nonzero_elements": nonzero,
        "total_elements": counts,
        "nonzero_fraction": {
            k: (float(nonzero[k]) / float(counts[k]) if counts[k] else 0.0) for k in groups
        },
    }


__all__ = [
    "BOCModel",
    "build_model",
    "count_module_calls",
    "gradient_report",
    "initialize_new_modules",
    "new_keys",
    "parameter_inventory",
]
