"""cscl-strong-nongnn-predictor-v1 — static triplet operator + enhanced Full-Y.

Protocol: ``configs/strong_nongnn_predictor_v1.yaml`` (frozen).

Scientific object
-----------------
A single, non-iterative, non-GNN high-order structural operator appended to the
frozen Full-Y computation:

* Triplet = wedge ``(i, j, k)``: centre ``j`` together with two **distinct**
  real chemical-bond neighbours ``i`` and ``k``.  ``(i, j, k)`` and
  ``(k, j, i)`` are the same triplet; every legal triplet is emitted exactly
  once (unordered endpoint pairs per centre).
* Real bonds are the unique undirected edges recovered from the frozen
  environment occurrence arrays (``env_bond_u/v/type``); on all 10 000 rows
  this set is bit-for-bit identical to ``pair_bucket == 0`` over
  ``pair_index`` (see the Phase-A audit).
* Features per triplet use the LIVE learned per-atom environment ``E``:
  a shared ``E -> 32`` projection, exchange-symmetric endpoint combination
  ``[h_i+h_k, |h_i-h_k|, h_i*h_k]``, the centre ``h_j``, symmetric bond-type
  sum/product over the two real bonds, the raw centre atom type (one-hot 28)
  and one fixed topology indicator (whether ``i`` and ``k`` are themselves
  bonded = triangle closure).
* The per-graph summary is ``[sum_t q_t ; sum_t q_t^2]`` (64 dims), appended
  to the original 814-dim Full-Y reader input.  The original reader input
  columns keep their historical parameter values; the 64 new columns are
  zero-initialised, so at initialisation the enhanced model computes exactly
  the frozen Full-Y function.

Non-GNN guarantee
-----------------
No neighbour hidden state is ever written back: ``E`` is produced once by the
(pre-existing) single-round static incidence aggregation, triplets are
evaluated once, and only their graph-level first/second moments enter the
reader.  There is no node/edge state update, no attention, no Transformer and
no iteration over graph structure.

Implementation notes
--------------------
The original forward (``AuditModel.forward`` with ``mask=C6_MASK``) is left
untouched.  The enhanced model only (a) replaces the reader with a wider-input
copy that preserves every old parameter value and zeroes the new columns and
(b) registers two hooks: one captures the environment tensor ``E`` produced by
the original fusion module, the other appends the triplet summary to the
reader input.  With the summary = 0 and the new reader columns = 0 the two
models are mathematically identical.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader

PROTOCOL_ID = "cscl-strong-nongnn-predictor-v1"

#: frozen triplet module dimensions (no width search in this round)
TRIPLET_PROJ_DIM = 32
TRIPLET_HIDDEN = 64
TRIPLET_OUT = 32
TRIPLET_SUMMARY_DIM = 2 * TRIPLET_OUT
#: frozen independent initialisation seed for the new module
TRIPLET_SEED = 20261011
#: fixed topology indicator count (triangle closure)
TRIPLET_FIXED_COUNT = 1

ATOM_CATEGORIES = int(p1.ATOM_CATEGORIES)  # 28
BOND_CATEGORIES = int(p1.BOND_CATEGORIES)  # 4

#: frozen label-free diagnostics switch names (analysis only, never training)
ABLATIONS = (
    "none",
    "zero_summary",
    "zero_endpoint_env",
    "zero_center_env",
    "zero_atom_type",
    "zero_closure",
    "zero_bond_types",
    "cyclic_bond_types",
)


# ---------------------------------------------------------------------------
# triplet construction (pure structure, no labels)
# ---------------------------------------------------------------------------


def unique_bonds_from_env(
    env_bond_u: torch.Tensor,
    env_bond_v: torch.Tensor,
    env_bond_type: torch.Tensor,
    num_nodes: int,
) -> dict[str, torch.Tensor]:
    """Deduplicate the frozen per-root bond occurrences into real bonds.

    Returns sorted-by-(lo, hi) unique bonds with their chemistry type and the
    global key used for closure lookups.
    """
    if env_bond_u.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=env_bond_u.device)
        return {"lo": empty, "hi": empty.clone(), "type": empty.clone(), "key": empty.clone()}
    lo = torch.minimum(env_bond_u, env_bond_v)
    hi = torch.maximum(env_bond_u, env_bond_v)
    key = lo * int(num_nodes) + hi
    order = torch.argsort(key)
    sorted_key = key[order]
    keep = torch.ones_like(sorted_key, dtype=torch.bool)
    keep[1:] = sorted_key[1:] != sorted_key[:-1]
    selection = order[keep]
    return {
        "lo": lo[selection],
        "hi": hi[selection],
        "type": env_bond_type[selection],
        "key": sorted_key[keep],
    }


def build_wedges(
    bonds: Mapping[str, torch.Tensor],
    num_nodes: int,
) -> dict[str, torch.Tensor]:
    """Enumerate every unique centre-two-bond wedge exactly once.

    For each centre ``j`` with ``d`` real bonds, all ``C(d, 2)`` unordered
    endpoint pairs are emitted; the enumeration is deterministic (half-edges
    sorted by ``(centre, neighbour)``) and independent of atom relabelling.
    """
    device = bonds["lo"].device
    lo, hi, btype = bonds["lo"], bonds["hi"], bonds["type"]
    if lo.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return {
            "center": empty,
            "end_i": empty.clone(),
            "end_k": empty.clone(),
            "bond_1": empty.clone(),
            "bond_2": empty.clone(),
            "closure": torch.zeros(0, dtype=torch.float32, device=device),
        }
    # both directions of every bond = half edges
    centers = torch.cat([lo, hi])
    neighbors = torch.cat([hi, lo])
    types = torch.cat([btype, btype])
    order = torch.argsort(centers * int(num_nodes) + neighbors)
    centers, neighbors, types = centers[order], neighbors[order], types[order]

    size = torch.bincount(centers, minlength=int(num_nodes))
    start = torch.zeros(int(num_nodes) + 1, dtype=torch.long, device=device)
    start[1:] = torch.cumsum(size, dim=0)
    local = torch.arange(centers.numel(), device=device) - start[centers]

    max_degree = int(size.max()) if size.numel() else 0
    first: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    for shift in range(1, max(max_degree, 1)):
        valid = (local + shift) < size[centers]
        idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if idx.numel():
            first.append(idx)
            second.append(idx + shift)
    if first:
        a = torch.cat(first)
        b = torch.cat(second)
    else:
        a = torch.zeros(0, dtype=torch.long, device=device)
        b = torch.zeros(0, dtype=torch.long, device=device)

    center = centers[a]
    end_i = neighbors[a]
    end_k = neighbors[b]
    bond_1 = types[a]
    bond_2 = types[b]

    # closed triangle indicator: are the two endpoints themselves bonded?
    closure = torch.zeros(end_i.numel(), dtype=torch.float32, device=device)
    if end_i.numel() and bonds["key"].numel():
        end_lo = torch.minimum(end_i, end_k)
        end_hi = torch.maximum(end_i, end_k)
        query = end_lo * int(num_nodes) + end_hi
        pos = torch.searchsorted(bonds["key"], query).clamp(max=int(bonds["key"].numel()) - 1)
        closure = (bonds["key"][pos] == query).to(torch.float32)
    return {
        "center": center,
        "end_i": end_i,
        "end_k": end_k,
        "bond_1": bond_1,
        "bond_2": bond_2,
        "closure": closure,
    }


def triplet_feature_dim() -> int:
    return (
        3 * TRIPLET_PROJ_DIM
        + TRIPLET_PROJ_DIM
        + BOND_CATEGORIES
        + BOND_CATEGORIES * BOND_CATEGORIES
        + ATOM_CATEGORIES
        + TRIPLET_FIXED_COUNT
    )


class StaticTripletOperator(nn.Module):
    """Shared non-iterative triplet function ``(E, structure) -> R^32``.

    The projection and the MLP are shared across every triplet and every
    molecule; no parameter is per-molecule or per-triplet.
    """

    def __init__(
        self,
        env_dim: int,
        *,
        proj_dim: int = TRIPLET_PROJ_DIM,
        hidden: int = TRIPLET_HIDDEN,
        out_dim: int = TRIPLET_OUT,
        atom_categories: int = ATOM_CATEGORIES,
        bond_categories: int = BOND_CATEGORIES,
        seed: int = TRIPLET_SEED,
    ) -> None:
        super().__init__()
        self.env_dim = int(env_dim)
        self.proj_dim = int(proj_dim)
        self.out_dim = int(out_dim)
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        in_dim = (
            3 * self.proj_dim
            + self.proj_dim
            + self.bond_categories
            + self.bond_categories * self.bond_categories
            + self.atom_categories
            + TRIPLET_FIXED_COUNT
        )
        self.in_dim = int(in_dim)
        self.env_proj = nn.Linear(self.env_dim, self.proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), self.out_dim),
        )
        self.reset_parameters(int(seed))

    def reset_parameters(self, seed: int) -> None:
        """Private-stream init identical in shape convention to ``nn.Linear``."""
        generator = torch.Generator().manual_seed(int(seed))
        for module in (self.env_proj, *self.mlp):
            if isinstance(module, nn.Linear):
                fan_in = int(module.weight.shape[1])
                bound = 1.0 / math.sqrt(float(max(fan_in, 1)))
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound, generator=generator)
                    if module.bias is not None:
                        module.bias.uniform_(-bound, bound, generator=generator)

    def features(
        self,
        E: torch.Tensor,
        center: torch.Tensor,
        end_i: torch.Tensor,
        end_k: torch.Tensor,
        bond_1: torch.Tensor,
        bond_2: torch.Tensor,
        atom_type_j: torch.Tensor,
        closure: torch.Tensor,
        *,
        ablation: str = "none",
    ) -> torch.Tensor:
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown triplet ablation {ablation!r}")
        h = self.env_proj(E)
        if ablation == "zero_endpoint_env":
            hi = torch.zeros((end_i.numel(), self.proj_dim), dtype=h.dtype, device=h.device)
            hk = torch.zeros_like(hi)
        else:
            hi = h[end_i]
            hk = h[end_k]
        if ablation == "zero_center_env":
            hj = torch.zeros((center.numel(), self.proj_dim), dtype=h.dtype, device=h.device)
        else:
            hj = h[center]

        if ablation == "zero_bond_types":
            b1 = torch.zeros_like(bond_1)
            b2 = torch.zeros_like(bond_2)
        elif ablation == "cyclic_bond_types":
            b1 = (bond_1 + 1) % self.bond_categories
            b2 = (bond_2 + 1) % self.bond_categories
        else:
            b1, b2 = bond_1, bond_2

        endpoint = torch.cat([hi + hk, torch.abs(hi - hk), hi * hk], dim=1)
        one_1 = torch.nn.functional.one_hot(b1, num_classes=self.bond_categories).to(h.dtype)
        one_2 = torch.nn.functional.one_hot(b2, num_classes=self.bond_categories).to(h.dtype)
        bond_pair = torch.cat(
            [
                one_1 + one_2,
                (
                    one_1.unsqueeze(2) * one_2.unsqueeze(1)
                    + one_2.unsqueeze(2) * one_1.unsqueeze(1)
                ).reshape(-1, self.bond_categories ** 2),
            ],
            dim=1,
        )
        if ablation == "zero_atom_type":
            atom = torch.zeros(
                (center.numel(), self.atom_categories), dtype=h.dtype, device=h.device
            )
        else:
            atom = torch.nn.functional.one_hot(
                atom_type_j, num_classes=self.atom_categories
            ).to(h.dtype)
        fixed = closure.unsqueeze(1)
        if ablation == "zero_closure":
            fixed = torch.zeros_like(fixed)
        return torch.cat([endpoint, hj, bond_pair, atom, fixed], dim=1)

    def forward(
        self,
        E: torch.Tensor,
        center: torch.Tensor,
        end_i: torch.Tensor,
        end_k: torch.Tensor,
        bond_1: torch.Tensor,
        bond_2: torch.Tensor,
        atom_type_j: torch.Tensor,
        closure: torch.Tensor,
        *,
        ablation: str = "none",
    ) -> torch.Tensor:
        features = self.features(
            E, center, end_i, end_k, bond_1, bond_2, atom_type_j, closure, ablation=ablation
        )
        if features.numel() == 0:
            return torch.zeros((0, self.out_dim), dtype=E.dtype, device=E.device)
        return self.mlp(features)


def pool_triplet_moments(
    q: torch.Tensor, triple_graph: torch.Tensor, n_graphs: int
) -> torch.Tensor:
    """Per-graph ``[sum q ; sum q^2]``; zero for molecules with no triplet."""
    out = int(q.shape[1]) if q.dim() > 1 else TRIPLET_OUT
    total = torch.zeros((int(n_graphs), out), dtype=q.dtype, device=q.device)
    squared = torch.zeros_like(total)
    if q.numel():
        index = triple_graph.long()
        total.index_add_(0, index, q)
        squared.index_add_(0, index, q * q)
    return torch.cat([total, squared], dim=1)


def triplet_module_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for name, p in model.named_parameters() if name.startswith("triplet_op.")))


# ---------------------------------------------------------------------------
# enhanced model: untouched original forward + wider reader + hooks
# ---------------------------------------------------------------------------


class StaticTripletFullY(sc.LatentScaleSEM108):
    """Full-Y with a static triplet summary appended to the reader input.

    * ``super().__init__`` builds the **exact** historical Full-Y
      (same parameter names, same RNG stream, same values).
    * The reader is replaced by a wider-input copy whose first
      ``spec.reader_input`` input columns are bit-identical to the original
      reader and whose new ``TRIPLET_SUMMARY_DIM`` columns are zero.
    * ``forward`` delegates to the untouched original forward; two hooks
      capture ``E`` and append the triplet summary at the reader boundary.
    """

    def __init__(
        self,
        config: Any,
        dictionary,
        *,
        subspace,
        spec: sc.ScaleSpec = sc.FULL,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        scale_seed: int = 0,
        triplet_seed: int = TRIPLET_SEED,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            spec=spec,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
            scale_seed=scale_seed,
        )
        # the new module must not perturb the post-construction global RNG
        # stream: with this restore the original model's dropout stream (and
        # every other global draw) continues exactly where the untouched
        # historical construction left it.
        rng_after_parent = torch.get_rng_state()
        self.scale_spec = spec
        self.triplet_summary_dim = TRIPLET_SUMMARY_DIM
        self.reader_input_original = int(spec.reader_input)
        self.triplet_op = StaticTripletOperator(int(spec.d), seed=int(triplet_seed))
        self._replace_reader(spec)
        torch.set_rng_state(rng_after_parent)
        #: analysis-only label-free diagnostics switch (never used in training)
        self.triplet_ablation = "none"
        #: last computed triplet tensors (diagnostics / provenance only)
        self.last_triplet: dict[str, torch.Tensor] | None = None
        self._captured_E: torch.Tensor | None = None
        self._forward_data: Any = None
        self._capture_provenance = False
        self._register_hooks()

    # -- construction ----------------------------------------------------------

    def _replace_reader(self, spec: sc.ScaleSpec) -> None:
        old = self.reader
        widened = int(spec.reader_input) + TRIPLET_SUMMARY_DIM
        new = GenericReader(widened, (spec.reader_hidden, spec.reader_hidden))
        with torch.no_grad():
            new.net[0].weight.zero_()
            new.net[0].weight[:, : int(spec.reader_input)].copy_(old.net[0].weight)
            new.net[0].bias.copy_(old.net[0].bias)
            for old_index, new_index in ((2, 2), (4, 4)):
                new.net[new_index].weight.copy_(old.net[old_index].weight)
                new.net[new_index].bias.copy_(old.net[old_index].bias)
        self.reader = new

    def _register_hooks(self) -> None:
        # capture the environment tensor produced by the original fusion MLP
        def _capture_E(_module, _inputs, output):
            self._captured_E = output

        self._capture_E_handle = self.fusion.register_forward_hook(_capture_E)

        # append the triplet summary at the original reader-input boundary
        def _append_summary(_module, inputs):
            unified = inputs[0]
            if self._captured_E is None:
                raise RuntimeError("triplet hook fired before the environment was produced")
            summary = self._triplet_summary()
            return (torch.cat([unified, summary], dim=1),)

        self._reader_hook = self.reader.register_forward_pre_hook(_append_summary)

    # -- forward ----------------------------------------------------------------

    def forward(self, data: Any, **kwargs: Any):
        self._forward_data = data
        try:
            return super().forward(data, **kwargs)
        finally:
            self._forward_data = None
            self._captured_E = None

    # -- triplet path -------------------------------------------------------------

    def _triplet_summary(self) -> torch.Tensor:
        data = self._forward_data
        E = self._captured_E
        device = E.device
        n_graphs = int(data.global_context.shape[0])
        num_nodes = int(E.shape[0])
        bonds = unique_bonds_from_env(
            data.env_bond_u.to(device), data.env_bond_v.to(device), data.env_bond_type.to(device), num_nodes
        )
        wedges = build_wedges(bonds, num_nodes)
        q = self.triplet_op(
            E,
            wedges["center"],
            wedges["end_i"],
            wedges["end_k"],
            wedges["bond_1"],
            wedges["bond_2"],
            data.dict_atom[wedges["center"]].to(device),
            wedges["closure"],
            ablation=self.triplet_ablation,
        )
        triple_graph = data.batch[wedges["center"]].long()
        summary = pool_triplet_moments(q, triple_graph, n_graphs)
        if self.triplet_ablation == "zero_summary":
            summary = torch.zeros_like(summary)
        self.last_triplet = {
            "q": q.detach(),
            "summary": summary.detach(),
            "n_triplets": int(wedges["center"].numel()),
        }
        if self._capture_provenance:
            self.last_triplet["provenance"] = {
                "center": wedges["center"].detach().cpu(),
                "end_i": wedges["end_i"].detach().cpu(),
                "end_k": wedges["end_k"].detach().cpu(),
                "bond_1": wedges["bond_1"].detach().cpu(),
                "bond_2": wedges["bond_2"].detach().cpu(),
                "closure": wedges["closure"].detach().cpu(),
                "triple_graph": triple_graph.detach().cpu(),
                "bond_lo": bonds["lo"].detach().cpu(),
                "bond_hi": bonds["hi"].detach().cpu(),
                "bond_type": bonds["type"].detach().cpu(),
                "E": E.detach().cpu(),
                "atom_type": data.dict_atom.detach().cpu(),
                "batch": data.batch.detach().cpu(),
                "q": q.detach().cpu(),
                "summary": summary.detach().cpu(),
            }
        return summary

    # -- helpers ------------------------------------------------------------------

    def reader_new_column_norm(self) -> float:
        weight = self.reader.net[0].weight
        return float(weight[:, self.reader_input_original :].detach().abs().sum().item())

    def shared_with_full_y(self, reference: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Bit-exact comparison of every parameter the two models share."""
        own = self.state_dict()
        mismatched: list[str] = []
        missing: list[str] = []
        for name, value in reference.items():
            if name not in own:
                missing.append(name)
                continue
            if name == "reader.net.0.weight":
                if not torch.equal(own[name][:, : self.reader_input_original], value):
                    mismatched.append(name)
            elif torch.is_tensor(value) and own[name].shape == value.shape:
                if not torch.equal(own[name], value):
                    mismatched.append(name)
            elif torch.is_tensor(value):
                mismatched.append(name)
        new_cols = own["reader.net.0.weight"][:, self.reader_input_original :]
        return {
            "missing_keys": missing,
            "mismatched_keys": mismatched,
            "new_reader_columns_all_zero": bool(torch.count_nonzero(new_cols) == 0),
            "identical_shared_parameters": bool(not missing and not mismatched and torch.count_nonzero(new_cols) == 0),
        }


def build_triplet_model(
    dictionary,
    seed: int,
    subspace,
    spec: sc.ScaleSpec = sc.FULL,
    *,
    clean_spec: Any = None,
    scale_seed: int = 0,
    triplet_seed: int = TRIPLET_SEED,
) -> StaticTripletFullY:
    """Build the enhanced model with the exact parent RNG stream."""
    clean_spec = sc.cssd.CSSD_SPEC if clean_spec is None else clean_spec
    torch.manual_seed(int(seed))
    return StaticTripletFullY(
        sc.cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        spec=spec,
        node_binding=clean_spec.node_binding,
        edge_binding=clean_spec.edge_binding,
        coding=clean_spec.coding,
        scale_seed=int(scale_seed),
        triplet_seed=int(triplet_seed),
    )


def load_full_y_state_into_triplet(
    model: StaticTripletFullY, state: Mapping[str, torch.Tensor]
) -> None:
    """Strictly load a historical Full-Y state (read-only replay/analysis)."""
    own = model.state_dict()
    for name, value in state.items():
        if name not in own:
            raise KeyError(f"unexpected key {name!r} in Full-Y state")
        if name == "reader.net.0.weight":
            own[name][:, : model.reader_input_original].copy_(value)
            own[name][:, model.reader_input_original :].zero_()
            continue
        if own[name].shape != value.shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(own[name].shape)} vs {tuple(value.shape)}")
        own[name].copy_(value)


def triplet_parameter_inventory(model: StaticTripletFullY) -> dict[str, int]:
    """Exact parameter accounting for the new branch."""
    reader_new = TRIPLET_SUMMARY_DIM * int(model.reader.net[0].weight.shape[0])
    return {
        "triplet_env_proj": int(model.triplet_op.env_proj.weight.numel() + model.triplet_op.env_proj.bias.numel()),
        "triplet_mlp": int(
            sum(
                p.numel()
                for name, p in model.triplet_op.mlp.named_parameters()
            )
        ),
        "reader_new_columns": int(reader_new),
        "triplet_total": int(triplet_module_parameters(model)),
        "full_model_parameters": int(sum(p.numel() for p in model.parameters())),
        "original_full_y_parameters": int(sc.FULL.parameters()),
    }


__all__ = [
    "PROTOCOL_ID",
    "TRIPLET_PROJ_DIM",
    "TRIPLET_HIDDEN",
    "TRIPLET_OUT",
    "TRIPLET_SUMMARY_DIM",
    "TRIPLET_SEED",
    "ATOM_CATEGORIES",
    "BOND_CATEGORIES",
    "ABLATIONS",
    "StaticTripletOperator",
    "StaticTripletFullY",
    "build_triplet_model",
    "build_wedges",
    "unique_bonds_from_env",
    "pool_triplet_moments",
    "triplet_feature_dim",
    "triplet_module_parameters",
    "triplet_parameter_inventory",
    "load_full_y_state_into_triplet",
]
