"""PEC-I1 — Pure Environment Composition, Static Composition Interface Audit.

Round: ``pec_i1``.  Pre-registration: ``notes/pec_i1_preregistration.md``.
Prior-artifact audit: ``notes/pec_i1_prior_artifact_audit.md``.

PEC-I1 imports PEC-v0's architecture unchanged and changes **exactly one**
scientific object: the graph-level composition / readout statistics interface.

* environment formation, roles, chemistry primitives, pair relation and pair
  composer are PEC-v0's, imported as-is (``pec_v0``);
* the unary readout becomes ``[mean(E), std(E), log1p(n)]`` (S0 moments);
* the pair readout becomes, per **audited S0 shortest-path distance bucket**
  ``d in {1,2,3,4,5+}``, ``[mean(c), std(c), log1p(n_d)]``;
* the reader hidden width is chosen deterministically so the total parameter
  count stays within 1 % of PEC-CD (no ``env_hidden`` / ``env_dim`` /
  ``pair_hidden`` / ``pair_dim`` change).

No message passing, no recurrence, no pair->centre, no mixed pair chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import pec_v0 as pec

# ---------------------------------------------------------------------------
# frozen interface constants
# ---------------------------------------------------------------------------

#: S0's audited pooling bucket count (``zinc_patch_path_pooling.DISTANCE_BUCKETS``
#: = 5, meaning ``1, 2, 3, 4, 5+``).  Reused verbatim; no new bucket is designed.
I1_DISTANCE_BUCKETS = 5

#: S0's audited ``mean_std`` convention: ``sqrt(clamp(var, min=0) + 1e-8)``.
I1_STD_EPS = 1.0e-8

I1_UNARY_WIDTH = 2 * pec.ENV_DIM + 1  # 97
I1_PAIR_BUCKET_WIDTH = 2 * pec.PAIR_DIM + 1  # 97

#: deterministic nearest-integer reader parameter match (pre-registration §4)
I1_READER_HIDDEN = 22


def i1_reader_input_width(
    env_dim: int = pec.ENV_DIM, pair_dim: int = pec.PAIR_DIM
) -> int:
    """Reader input width of the PEC-I1 pooling interface (590 by default)."""
    return int(
        2 * int(env_dim)
        + 1
        + I1_DISTANCE_BUCKETS * (2 * int(pair_dim) + 1)
        + pec.TOPO_GLOBAL_DIM
    )


# ---------------------------------------------------------------------------
# S0-audited statistical pooling (mean / population std / log1p count)
# ---------------------------------------------------------------------------


def pool_mean_std_count(
    values: torch.Tensor,
    index: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """``[mean, std, log1p(count)]`` per group, exactly S0's ``mean_std`` mode.

    Empty groups follow the S0 convention: ``mean = 0``,
    ``std = sqrt(0 + 1e-8) = 1e-4``, ``log1p(0) = 0``.
    """
    n_groups = int(n_groups)
    width = int(values.shape[1])
    total = values.new_zeros((n_groups, width))
    counts = values.new_zeros((n_groups, 1))
    if values.numel():
        total.index_add_(0, index, values)
        counts.index_add_(
            0, index, torch.ones((index.shape[0], 1), device=values.device, dtype=values.dtype)
        )
    mean = total / counts.clamp_min(1.0)
    squared = values.new_zeros((n_groups, width))
    if values.numel():
        squared.index_add_(0, index, values * values)
    variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
    std = torch.sqrt(variance + I1_STD_EPS)
    return torch.cat([mean, std, torch.log1p(counts)], dim=1)


def pair_bucket_from_rho(
    pair_rho: torch.Tensor, n_buckets: int = I1_DISTANCE_BUCKETS
) -> torch.Tensor:
    """S0 distance bucket index from PEC's frozen 8-class distance one-hot.

    PEC's ``pair_rho`` begins with ``pec.DISTANCE_BUCKETS`` (= 8) distance
    one-hot columns whose index is ``min(max(d,1),8)-1``.  The S0 audited
    bucket is ``min(max(d,1),5)-1 = min(index_8, 4)``.  This is a deterministic
    re-index, not a new feature.
    """
    if pair_rho.shape[1] < pec.DISTANCE_BUCKETS:
        raise RuntimeError(
            f"pair_rho width {pair_rho.shape[1]} < {pec.DISTANCE_BUCKETS}"
        )
    index_8 = pair_rho[:, : pec.DISTANCE_BUCKETS].argmax(dim=1)
    return index_8.clamp_max(int(n_buckets) - 1)


def pool_pairs_by_bucket(
    values: torch.Tensor,
    pair_batch: torch.Tensor,
    pair_bucket: torch.Tensor,
    n_graphs: int,
    n_buckets: int = I1_DISTANCE_BUCKETS,
) -> torch.Tensor:
    """Concatenate ``[mean, std, log1p(count)]`` over S0 distance buckets."""
    blocks: list[torch.Tensor] = []
    for bucket in range(int(n_buckets)):
        mask = pair_bucket == int(bucket)
        blocks.append(
            pool_mean_std_count(values[mask], pair_batch[mask], int(n_graphs))
        )
    return torch.cat(blocks, dim=1)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class PECI1Model(pec.PECModel):
    """PEC-v0 architecture with the S0-style statistical pooling interface.

    Only ``forward`` (pooling + reader input) and the reader width differ from
    ``pec.PECModel``.  ``environment`` and ``pair`` are constructed by the
    parent with the same shapes and the same RNG order, so with an identical
    seed their parameters are bit-identical to a PEC-CD model.
    """

    def __init__(
        self,
        *,
        role_mode: str = "dense",
        d_node: np.ndarray | None = None,
        d_edge: np.ndarray | None = None,
        env_hidden: int = pec.ENV_HIDDEN,
        env_dim: int = pec.ENV_DIM,
        pair_hidden: int = pec.PAIR_HIDDEN,
        pair_dim: int = pec.PAIR_DIM,
        reader_hidden: int = I1_READER_HIDDEN,
        ablation: str = "true",
    ) -> None:
        super().__init__(
            role_mode=role_mode,
            d_node=d_node,
            d_edge=d_edge,
            env_hidden=env_hidden,
            env_dim=env_dim,
            pair_hidden=pair_hidden,
            pair_dim=pair_dim,
            reader_hidden=int(reader_hidden),
            ablation=ablation,
        )
        self.i1_reader_input_width = i1_reader_input_width(env_dim, pair_dim)
        # The parent reader has PEC's input width; rebuild it at the I1 width.
        # ``environment`` / ``pair`` were already drawn, so their parameters are
        # unaffected by this replacement.
        self.reader = nn.Sequential(
            nn.Linear(self.i1_reader_input_width, int(reader_hidden)),
            nn.SiLU(),
            nn.Linear(int(reader_hidden), 1),
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        shuffle_generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        n_graphs = int(batch["n_graphs"].item())
        environments = self.encode_environments(batch)

        if self.ablation == "shuffle":
            environments = pec._shuffle_within_graphs(
                environments, batch["node_graph"], n_graphs, shuffle_generator
            )

        unary = pool_mean_std_count(
            environments, batch["node_graph"], n_graphs
        )

        pairs = self.compose_pairs(environments, batch)
        bucket = pair_bucket_from_rho(batch["pair_rho"])
        pair_pooled = pool_pairs_by_bucket(
            pairs, batch["pair_graph"], bucket, n_graphs
        )
        if self.ablation == "bag":
            pair_pooled = torch.zeros_like(pair_pooled)

        reader_input = torch.cat(
            [unary, pair_pooled, batch["global_topo"]], dim=1
        )
        if reader_input.shape[1] != self.i1_reader_input_width:
            raise RuntimeError(
                f"I1 reader input width {reader_input.shape[1]} "
                f"!= {self.i1_reader_input_width}"
            )
        prediction = self.reader(reader_input).squeeze(-1)
        return {
            "prediction": prediction,
            "environments": environments,
            "pairs": pairs,
            "pair_bucket": bucket,
        }


def build_i1_model(
    role_mode: str = "dense",
    *,
    d_node: np.ndarray | None = None,
    d_edge: np.ndarray | None = None,
    env_hidden: int = pec.ENV_HIDDEN,
    pair_hidden: int = pec.PAIR_HIDDEN,
    reader_hidden: int = I1_READER_HIDDEN,
    ablation: str = "true",
    seed: int = 0,
) -> PECI1Model:
    """Build one PEC-I1 arm; mirrors ``pec_v0.build_model`` seeding order."""
    torch.manual_seed(int(seed))
    return PECI1Model(
        role_mode=role_mode,
        d_node=d_node,
        d_edge=d_edge,
        env_hidden=int(env_hidden),
        env_dim=pec.ENV_DIM,
        pair_hidden=int(pair_hidden),
        pair_dim=pec.PAIR_DIM,
        reader_hidden=int(reader_hidden),
        ablation=ablation,
    )


# ---------------------------------------------------------------------------
# parameter matching
# ---------------------------------------------------------------------------


def i1_reader_params(reader_hidden: int, input_width: int | None = None) -> int:
    width = i1_reader_input_width() if input_width is None else int(input_width)
    return (width + 2) * int(reader_hidden) + 1


def i1_non_reader_params(
    *,
    role_mode: str = "dense",
    env_hidden: int = pec.ENV_HIDDEN,
    env_dim: int = pec.ENV_DIM,
    pair_hidden: int = pec.PAIR_HIDDEN,
    pair_dim: int = pec.PAIR_DIM,
) -> int:
    width = i1_reader_input_width(env_dim, pair_dim)
    probe = PECI1Model(
        role_mode=role_mode,
        env_hidden=int(env_hidden),
        env_dim=int(env_dim),
        pair_hidden=int(pair_hidden),
        pair_dim=int(pair_dim),
        reader_hidden=1,
    )
    return pec.n_params(probe) - i1_reader_params(1, width)


def match_reader_hidden(
    target_total: int,
    *,
    role_mode: str = "dense",
    env_hidden: int = pec.ENV_HIDDEN,
    env_dim: int = pec.ENV_DIM,
    pair_hidden: int = pec.PAIR_HIDDEN,
    pair_dim: int = pec.PAIR_DIM,
) -> int:
    """Nearest-integer reader hidden width matching ``target_total`` exactly.

    Pre-committed rule (pre-registration §4): minimise
    ``|params(CD-I1) - target_total|``; ties resolve to the smaller width.
    """
    width = i1_reader_input_width(env_dim, pair_dim)
    base = i1_non_reader_params(
        role_mode=role_mode,
        env_hidden=env_hidden,
        env_dim=env_dim,
        pair_hidden=pair_hidden,
        pair_dim=pair_dim,
    )
    approx = (int(target_total) - base - 1) / (width + 2)
    candidates = {
        candidate
        for candidate in range(max(1, int(np.floor(approx)) - 3), int(np.ceil(approx)) + 4)
        if candidate >= 1
    }
    return min(
        candidates,
        key=lambda candidate: (
            abs(base + (width + 2) * candidate + 1 - int(target_total)),
            candidate,
        ),
    )


def parameter_accounting(
    target_total: int = 94049,
    *,
    role_mode: str = "dense",
) -> dict[str, Any]:
    """Report baseline/candidate counts and the chosen reader width."""
    from tracks.ksvd.experiments.luyin16 import pec_v0_gate0 as _g0

    d_node, d_edge = _g0._dictionary(seed=0)
    baseline = pec.build_model(
        role_mode, d_node=d_node if role_mode != "coarse" else None,
        d_edge=d_edge if role_mode != "coarse" else None, seed=0,
    )
    baseline_params = pec.n_params(baseline)
    reader_hidden = match_reader_hidden(baseline_params, role_mode=role_mode)
    candidate = build_i1_model(
        role_mode=role_mode,
        d_node=d_node if role_mode != "coarse" else None,
        d_edge=d_edge if role_mode != "coarse" else None,
        reader_hidden=reader_hidden,
        seed=0,
    )
    candidate_params = pec.n_params(candidate)
    absolute = candidate_params - baseline_params
    return {
        "role_mode": role_mode,
        "baseline_total_params": int(baseline_params),
        "candidate_total_params": int(candidate_params),
        "absolute_diff": int(absolute),
        "relative_diff": float(absolute) / float(baseline_params),
        "reader_hidden": int(reader_hidden),
        "reader_input_width": int(i1_reader_input_width()),
        "baseline_reader_hidden": int(pec.READER_HIDDEN),
        "candidate_reader_params": int(i1_reader_params(reader_hidden)),
        "non_reader_params": int(
            i1_non_reader_params(role_mode=role_mode)
        ),
        "tolerance": 0.01,
        "within_tolerance": abs(float(absolute)) / float(baseline_params) <= 0.01,
    }


# ---------------------------------------------------------------------------
# weight-sharing / correctness helpers
# ---------------------------------------------------------------------------

_SHARED_PREFIXES = ("environment.", "pair.")


def shared_environment_state_keys(model: nn.Module) -> list[str]:
    return [
        key
        for key, _value in model.state_dict().items()
        if key.startswith(_SHARED_PREFIXES)
    ]


def environment_pair_max_abs_diff(
    old: pec.PECModel, new: PECI1Model
) -> tuple[float, float]:
    """Bit-level comparison of the shared ``environment``/``pair`` parameters."""
    old_state = old.state_dict()
    new_state = new.state_dict()
    env_diff = 0.0
    pair_diff = 0.0
    for key in old_state:
        if not key.startswith(_SHARED_PREFIXES):
            continue
        delta = float((old_state[key] - new_state[key]).abs().max())
        if key.startswith("environment."):
            env_diff = max(env_diff, delta)
        else:
            pair_diff = max(pair_diff, delta)
    return env_diff, pair_diff


# ---------------------------------------------------------------------------
# Stage A — zero-training Patch-B recoverability audit
# ---------------------------------------------------------------------------

#: float tolerance counted as an exact recovery (float32 round-off headroom)
EXACT_TOL = 1.0e-5

#: S0 shell-pair taxonomy for radius 2 (6 classes, includes the empty (0,0))
S0_SHELL_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (2, 2),
)

BLOCK_NAMES = ("atom_shell", "bond_shell", "root_atom", "incident", "scalars")


@dataclass
class BlockAccumulator:
    name: str
    dim: int
    max_abs_error: float = 0.0
    sum_abs_error: float = 0.0
    count: int = 0
    exact: int = 0
    unexplained: int = 0
    target_rows: list[np.ndarray] = field(default_factory=list)
    recovered_rows: list[np.ndarray] = field(default_factory=list)

    def update(self, recovered: np.ndarray, target: np.ndarray) -> None:
        error = np.abs(
            recovered.astype(np.float64) - target.astype(np.float64)
        ).reshape(-1)
        self.max_abs_error = max(self.max_abs_error, float(error.max(initial=0.0)))
        self.sum_abs_error += float(error.sum())
        self.count += int(error.size)
        self.exact += int((error <= EXACT_TOL).sum())
        self.unexplained += int((error > EXACT_TOL).sum())
        self.target_rows.append(np.asarray(target, dtype=np.float64).reshape(-1))
        self.recovered_rows.append(np.asarray(recovered, dtype=np.float64).reshape(-1))

    def to_json(self) -> dict[str, Any]:
        target = (
            np.stack(self.target_rows, axis=0) if self.target_rows else np.zeros((0, self.dim))
        )
        recovered = (
            np.stack(self.recovered_rows, axis=0)
            if self.recovered_rows
            else np.zeros((0, self.dim))
        )
        rank_target = int(np.linalg.matrix_rank(target)) if target.size else 0
        rank_recovered = int(np.linalg.matrix_rank(recovered)) if recovered.size else 0
        return {
            "block": self.name,
            "dim": int(self.dim),
            "coordinates": int(self.count),
            "max_abs_error": float(self.max_abs_error),
            "mean_abs_error": float(self.sum_abs_error / max(self.count, 1)),
            "exact_fraction": float(self.exact / max(self.count, 1)),
            "unexplained_coordinates": int(self.unexplained),
            "rank_target": rank_target,
            "rank_recovered": rank_recovered,
            "row_count": int(target.shape[0]),
        }


def _s0_blocks(graph: Any, center: int, node_types: np.ndarray, edge_types: Mapping[tuple[int, int], int]):
    """Return S0 ``patch_cont`` blocks for one rooted radius-2 patch."""
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

    distances = zpp._ego_distances(graph, int(center), pec.PATCH_RADIUS)
    descriptor, nodes, boundary = zpp._shell_descriptor(
        graph,
        int(center),
        node_types,
        edge_types,
        distances,
        pec.PATCH_RADIUS,
    )
    descriptor = np.asarray(descriptor, dtype=np.float64)
    shell_pairs = zpp._shell_pairs_for_radius(pec.PATCH_RADIUS)
    atom_dim = (pec.PATCH_RADIUS + 1) * pec.ATOM_CATEGORIES
    bond_dim = len(shell_pairs) * pec.BOND_CATEGORIES
    offset = 0
    atom_shell = descriptor[offset : offset + atom_dim].reshape(
        pec.PATCH_RADIUS + 1, pec.ATOM_CATEGORIES
    )
    offset += atom_dim
    bond_shell = descriptor[offset : offset + bond_dim].reshape(
        len(shell_pairs), pec.BOND_CATEGORIES
    )
    offset += bond_dim
    root_atom = descriptor[offset : offset + pec.ATOM_CATEGORIES]
    offset += pec.ATOM_CATEGORIES
    incident = descriptor[offset : offset + pec.BOND_CATEGORIES]
    offset += pec.BOND_CATEGORIES
    scalars = descriptor[offset : offset + pec.TOPO_ROOT_DIM]
    return {
        "atom_shell": atom_shell,
        "bond_shell": bond_shell,
        "root_atom": root_atom,
        "incident": incident,
        "scalars": scalars,
        "nodes": nodes,
        "boundary": boundary,
        "distances": distances,
    }


def _pec_coarse_primitives(sample: pec.MoleculeSample) -> dict[str, np.ndarray]:
    """Coarse PEC environment primitives from the cached occurrence record.

    ``binding_v[r, s, a]`` = number of patch occurrences for root ``r`` in
    shell ``s`` with atom category ``a`` (the shell-one-hot block of the PEC
    dense node binding).  ``binding_e[r, sp, b]`` is the analogous shell-pair
    block of the edge binding.
    """
    n = int(sample.n_nodes)
    binding_v = np.zeros((n, pec.SHELL_CLASSES, pec.ATOM_CATEGORIES), dtype=np.float64)
    np.add.at(
        binding_v,
        (
            sample.occ_root,
            sample.occ_shell,
            sample.atom_idx[sample.occ_node],
        ),
        1.0,
    )
    binding_e = np.zeros(
        (n, pec.SHELLPAIR_CLASSES, pec.BOND_CATEGORIES), dtype=np.float64
    )
    np.add.at(
        binding_e,
        (
            sample.eocc_root,
            sample.eocc_shellpair,
            sample.bond_idx[sample.eocc_edge],
        ),
        1.0,
    )
    return {"binding_v": binding_v, "binding_e": binding_e}


def recover_blocks_for_root(
    sample: pec.MoleculeSample,
    coarse: Mapping[str, np.ndarray],
    s0: Mapping[str, Any],
    root: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """Return ``{block: (recovered, s0_target, method)}`` for one root."""
    scalars = np.asarray(sample.root_scalars[root], dtype=np.float64)
    patch_n = float(np.expm1(scalars[0]))
    patch_m = float(np.expm1(scalars[1]))
    root_degree = float(scalars[2])
    molecule_mean_degree = float(scalars[3])

    # --- atom shell: coarse (shell x atom) counts / patch node count -------
    atom_shell = coarse["binding_v"][root] / max(patch_n, 1.0)

    # --- bond shell: coarse (shellpair x bond) counts / patch edge count ---
    # PEC's role one-hot covers the 5 non-(0,0) classes; (0,0) is unexpressible
    # and is reconstructed as the structurally empty block.
    bond_shell = np.zeros(
        (len(S0_SHELL_PAIRS), pec.BOND_CATEGORIES), dtype=np.float64
    )
    for class_index, shell_pair in enumerate(S0_SHELL_PAIRS):
        if shell_pair == (0, 0):
            continue
        pec_index = pec.SHELLPAIRS.index(shell_pair)
        bond_shell[class_index] = coarse["binding_e"][root, pec_index] / max(patch_m, 1.0)

    # --- root atom ---------------------------------------------------------
    root_atom = np.zeros(pec.ATOM_CATEGORIES, dtype=np.float64)
    root_atom[int(sample.atom_idx[root])] = 1.0

    # --- incident bond composition: (0,1) counts / known root degree -------
    incident = coarse["binding_e"][root, pec.shellpair_index(0, 1)] / max(root_degree, 1.0)

    # --- structural scalars (S0 order), exact known-denominator transforms --
    patch_cycle_rank = max(patch_m - patch_n + 1.0, 0.0)
    recovered_scalars = np.asarray(
        [
            scalars[0],  # log1p(patch n_nodes)
            scalars[1],  # log1p(patch n_edges)
            scalars[4],  # boundary fraction
            patch_cycle_rank / max(patch_n, 1.0),
            root_degree / 4.0,
            molecule_mean_degree / 4.0,
        ],
        dtype=np.float64,
    )

    return {
        "atom_shell": (atom_shell, s0["atom_shell"], "analytic coarse binding / n_nodes"),
        "bond_shell": (bond_shell, s0["bond_shell"], "analytic coarse binding / n_edges"),
        "root_atom": (root_atom, s0["root_atom"], "analytic root chemistry one-hot"),
        "incident": (incident, s0["incident"], "analytic (0,1) binding / root degree"),
        "scalars": (recovered_scalars, s0["scalars"], "analytic known-denominator transform"),
    }


def audit_molecule(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    sample: pec.MoleculeSample,
    accumulators: Mapping[str, BlockAccumulator],
    *,
    root_limit: int | None = None,
) -> dict[str, int]:
    """Update block accumulators for every root of one molecule."""
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

    coarse = _pec_coarse_primitives(sample)
    counts = {"roots": 0, "s0_00_edges": 0, "s0_02_edges": 0}
    roots = range(int(sample.n_nodes))
    if root_limit is not None:
        roots = range(min(int(root_limit), int(sample.n_nodes)))
    for root in roots:
        s0 = _s0_blocks(graph, int(root), node_types, edge_types)
        # record S0's structurally empty shell-pair classes
        distances = s0["distances"]
        shell_pairs = zpp._shell_pairs_for_radius(pec.PATCH_RADIUS)
        induced = graph.induced(set(s0["nodes"]))
        for left, right in induced.edges():
            pair = tuple(
                sorted(
                    (
                        int(distances[int(left)]),
                        int(distances[int(right)]),
                    )
                )
            )
            if pair == (0, 0):
                counts["s0_00_edges"] += 1
            if pair == (0, 2):
                counts["s0_02_edges"] += 1
        del shell_pairs
        s0 = dict(s0)
        recovered = recover_blocks_for_root(sample, coarse, s0, int(root))
        for name, (value, target, _method) in recovered.items():
            accumulators[name].update(value, target)
        counts["roots"] += 1
    return counts


def _rank_summary(accumulator: BlockAccumulator) -> dict[str, Any]:
    payload = accumulator.to_json()
    return payload


def stage_a_recoverability(
    molecules: Sequence[
        tuple[Any, np.ndarray, Mapping[tuple[int, int], int], pec.MoleculeSample]
    ],
    *,
    root_limit: int | None = None,
) -> dict[str, Any]:
    """Label-free recoverability audit over ``(graph, atom_types, edge_types, sample)``.

    No target ``y`` is read anywhere.
    """
    accumulators = {
        "atom_shell": BlockAccumulator(
            "atom_shell", (pec.PATCH_RADIUS + 1) * pec.ATOM_CATEGORIES
        ),
        "bond_shell": BlockAccumulator(
            "bond_shell", len(S0_SHELL_PAIRS) * pec.BOND_CATEGORIES
        ),
        "root_atom": BlockAccumulator("root_atom", pec.ATOM_CATEGORIES),
        "incident": BlockAccumulator("incident", pec.BOND_CATEGORIES),
        "scalars": BlockAccumulator("scalars", pec.TOPO_ROOT_DIM),
    }
    totals = {"molecules": 0, "roots": 0, "s0_00_edges": 0, "s0_02_edges": 0}
    for graph, node_types, edge_types, sample in molecules:
        counts = audit_molecule(
            graph,
            node_types,
            edge_types,
            sample,
            accumulators,
            root_limit=root_limit,
        )
        totals["molecules"] += 1
        totals["roots"] += counts["roots"]
        totals["s0_00_edges"] += counts["s0_00_edges"]
        totals["s0_02_edges"] += counts["s0_02_edges"]
    blocks = {name: accumulator.to_json() for name, accumulator in accumulators.items()}
    return {
        "protocol_version": "pec_i1",
        "official_test_loaded": False,
        "stage": "A_zero_training_patch_b_recoverability",
        "reads_target_y": False,
        "exact_tolerance": EXACT_TOL,
        "totals": totals,
        "blocks": blocks,
        "s0_shell_pairs": [list(pair) for pair in S0_SHELL_PAIRS],
        "pec_shell_pairs": [list(pair) for pair in pec.SHELLPAIRS],
        "missing_shell_pair_class": [0, 0],
        "shell_pair_note": (
            "S0 declares 6 shell-pair classes including (0,0); PEC's role "
            "one-hot covers 5 and cannot express (0,0). (0,0) is structurally "
            "empty (shell 0 is the singleton root, so a (0,0) induced bond "
            "would be a self-loop). (0,2) is expressible in PEC (index 1) and "
            "is likewise structurally empty."
        ),
    }


# ---------------------------------------------------------------------------
# correctness gates G0-G8
# ---------------------------------------------------------------------------

from tracks.ksvd.experiments.luyin16.pec_v0_gate0 import (  # noqa: E402
    _batch,
    _chain,
    _chem,
    _dictionary,
    _ring,
    _sample,
)


def check_g0_environment_equivalence(seed: int = 0) -> dict[str, Any]:
    d_node, d_edge = _dictionary(seed=0)
    old = pec.build_model("dense", d_node=d_node, d_edge=d_edge, seed=seed)
    new = build_i1_model("dense", d_node=d_node, d_edge=d_edge, seed=seed, reader_hidden=I1_READER_HIDDEN)
    env_diff, pair_diff = environment_pair_max_abs_diff(old, new)
    batch = _batch([_sample(_ring(6)), _sample(_chain(5))])
    old_env = old.encode_environments(batch)
    new_env = new.encode_environments(batch)
    env_out = float((old_env - new_env).abs().max())
    old_pairs = old.compose_pairs(old_env, batch)
    new_pairs = new.compose_pairs(new_env, batch)
    pair_out = float((old_pairs - new_pairs).abs().max())
    assert env_diff == 0.0 and pair_diff == 0.0, (env_diff, pair_diff)
    assert env_out == 0.0 and pair_out == 0.0, (env_out, pair_out)
    return {
        "environment_param_max_abs_diff": env_diff,
        "pair_param_max_abs_diff": pair_diff,
        "environment_output_max_abs_diff": env_out,
        "pair_output_max_abs_diff": pair_out,
        "verdict": "PASS",
    }


def check_g1_pair_equivalence(seed: int = 0) -> dict[str, Any]:
    """The only change is after c_ij exists; c_ij itself is identical."""
    result = check_g0_environment_equivalence(seed=seed)
    return {"pair_output_max_abs_diff": result["pair_output_max_abs_diff"], "verdict": "PASS"}


def check_g2_no_message_passing(seed: int = 0) -> dict[str, Any]:
    model = build_i1_model("dense", d_node=_dictionary(seed=0)[0], d_edge=_dictionary(seed=0)[1], seed=seed)
    batch = _batch([_sample(_ring(6)), _sample(_chain(4))])

    original = model.compose_pairs

    def _explode(*_args, **_kwargs):
        raise RuntimeError("pair->centre write-back attempted")

    model.compose_pairs = _explode  # type: ignore[method-assign]
    env_ok = model.encode_environments(batch)
    assert torch.isfinite(env_ok).all()
    forward_raised = False
    try:
        model(batch)
    except RuntimeError:
        forward_raised = True
    model.compose_pairs = original  # type: ignore[method-assign]
    assert forward_raised, "pair composer was not reached by the forward pass"
    forward = model(batch)
    assert torch.isfinite(forward["prediction"]).all()
    # environment formation never reads pair tensors
    env_a = model.encode_environments(batch)
    mutated = dict(batch)
    mutated["pair_rho"] = torch.randn_like(batch["pair_rho"])
    env_b = model.encode_environments(mutated)
    assert torch.equal(env_a, env_b)
    return {
        "environment_works_without_pair_composer": True,
        "forward_requires_pair_composer": True,
        "environment_bit_identical_under_pair_mutation": True,
        "verdict": "PASS",
    }


def check_g3_environment_freeze(seed: int = 0) -> dict[str, Any]:
    first = _sample(_ring(6), seed=1)
    second = _sample(_chain(5), seed=2)
    model = build_i1_model("dense", d_node=_dictionary(seed=0)[0], d_edge=_dictionary(seed=0)[1], seed=seed)
    batch = _batch([first, second])
    env_before = model.encode_environments(batch)
    chain = _chain(5)
    mutated = pec.collate(
        [
            first,
            pec.build_sample(
                chain,
                [9] * 5,
                {chain.edge_key(u, v): 3 for u, v in chain.edges()},
            ),
        ]
    )
    mutated["pair_rho"] = torch.randn_like(mutated["pair_rho"])
    env_after = model.encode_environments(mutated)
    n_first = int(first.n_nodes)
    assert torch.equal(env_before[:n_first], env_after[:n_first])
    return {"environment_freeze_bit_identical": True, "verdict": "PASS"}


def check_g4_chemistry_purity(seed: int = 0) -> dict[str, Any]:
    graph = _ring(6)
    base = _sample(graph, atom_categories=(0, 1, 2, 3), bond_category=1)
    changed = _sample(graph, atom_categories=(4, 5, 6, 7), bond_category=3)
    assert np.array_equal(base.node_basis, changed.node_basis)
    assert np.array_equal(base.edge_basis, changed.edge_basis)
    assert np.array_equal(base.pair_rho, changed.pair_rho)
    assert np.array_equal(base.occ_shell, changed.occ_shell)
    assert np.array_equal(base.eocc_shellpair, changed.eocc_shellpair)
    # chemistry placement must still move the environment
    atom_types, edge_types = _chem(graph, (0, 1, 2, 3), 1)
    perm = [3, 0, 5, 1, 4, 2]
    shuffled = pec.build_sample(
        graph, [atom_types[perm[node]] for node in graph.nodes], edge_types
    )
    model = build_i1_model("dense", d_node=_dictionary(seed=0)[0], d_edge=_dictionary(seed=0)[1], seed=seed)
    batch = _batch([base, shuffled])
    env = model.encode_environments(batch)
    n_first = int(base.n_nodes)
    delta = float((env[:n_first] - env[n_first:]).abs().max())
    assert delta > 0.0
    return {
        "topology_invariant_under_chemistry": True,
        "pair_relation_invariant_under_chemistry": True,
        "chemistry_placement_changes_environment": delta,
        "verdict": "PASS",
    }


def check_g5_relabel_invariance(seed: int = 0) -> dict[str, Any]:
    from tracks.ksvd.code.graph import from_edges

    graph = _ring(6)
    atom_types, edge_types = _chem(graph, (0, 1, 2, 3), 1)
    base = pec.build_sample(
        graph, [atom_types[node] for node in graph.nodes], edge_types
    )
    perm = [2, 0, 5, 3, 1, 4]
    inverse = {old: new for new, old in enumerate(perm)}
    relabelled = from_edges(len(perm), [(inverse[u], inverse[v]) for u, v in graph.edges()])
    relabelled_atoms = [0] * len(perm)
    for old, new in inverse.items():
        relabelled_atoms[new] = atom_types[old]
    relabelled_types = {
        relabelled.edge_key(u, v): edge_types[graph.edge_key(perm[u], perm[v])]
        for u, v in relabelled.edges()
    }
    moved = pec.build_sample(relabelled, relabelled_atoms, relabelled_types)
    model = build_i1_model("dense", d_node=_dictionary(seed=0)[0], d_edge=_dictionary(seed=0)[1], seed=seed)
    out_base = model(_batch([base]))["prediction"]
    out_moved = model(_batch([moved]))["prediction"]
    delta = float((out_base - out_moved).abs().max())
    assert delta < 1.0e-5, delta
    return {"relabel_prediction_max_abs_diff": delta, "verdict": "PASS"}


def check_g6_bucket_correctness() -> dict[str, Any]:
    """Toy buckets vs an independent numpy reference, incl. empty/5+ cases."""

    def _buckets_for(graph):
        atom_types, edge_types = _chem(graph, (0, 1, 2, 3), 1)
        sample = pec.build_sample(
            graph, [atom_types[node] for node in graph.nodes], edge_types
        )
        batch = pec.collate([sample])
        bucket = pair_bucket_from_rho(batch["pair_rho"])
        distances = [
            abs(j - i)
            for i in range(sample.n_nodes)
            for j in range(i + 1, sample.n_nodes)
        ]
        expected = np.asarray(
            [min(max(int(d), 1), I1_DISTANCE_BUCKETS) - 1 for d in distances],
            dtype=np.int64,
        )
        assert np.array_equal(bucket.numpy(), expected), (bucket.numpy(), expected)
        return sample, batch, bucket

    # chain(4): distances 1,2,3 -> buckets 0,1,2; buckets 3,4 empty
    _sample4, batch4, bucket4 = _buckets_for(_chain(4))
    counts4 = np.bincount(bucket4.numpy(), minlength=I1_DISTANCE_BUCKETS)
    assert counts4[3] == 0 and counts4[4] == 0
    # chain(11): max distance 10 -> the 5+ bucket is populated
    _sample11, batch11, bucket11 = _buckets_for(_chain(11))
    assert int((bucket11.numpy() == 4).sum()) > 0

    values = torch.arange(batch4["pair_left"].shape[0], dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    per_bucket = 2 * values.shape[1] + 1
    pooled = pool_pairs_by_bucket(values, batch4["pair_graph"], bucket4, 1)
    assert pooled.shape[1] == I1_DISTANCE_BUCKETS * per_bucket
    for d in range(I1_DISTANCE_BUCKETS):
        block = pooled[0, d * per_bucket : (d + 1) * per_bucket]
        mask = bucket4 == d
        current = values[mask]
        reference_mean = current.mean(dim=0) if current.numel() else torch.zeros(2)
        if current.numel():
            reference_std = torch.sqrt(current.var(dim=0, unbiased=False) + I1_STD_EPS)
        else:
            reference_std = torch.full((2,), float(np.sqrt(I1_STD_EPS)))
        assert torch.allclose(block[:2], reference_mean, atol=1e-6)
        assert torch.allclose(block[2:4], reference_std, atol=1e-6), (block[2:4], reference_std)
        assert abs(float(block[4]) - float(np.log1p(int(mask.sum())))) < 1e-6
    return {
        "bucket_reference_matches": True,
        "empty_bucket_std": float(np.sqrt(I1_STD_EPS)),
        "empty_bucket_counts_chain4": counts4.tolist(),
        "verdict": "PASS",
    }


def check_g7_gradients(seed: int = 0) -> dict[str, float]:
    model = build_i1_model("dense", d_node=_dictionary(seed=0)[0], d_edge=_dictionary(seed=0)[1], seed=seed)
    batch = _batch([_sample(_ring(6), seed=5), _sample(_chain(4), seed=6)])
    out = model(batch)
    loss = out["prediction"].pow(2).mean()
    loss.backward()
    report: dict[str, float] = {}
    for name, parameter in [
        ("m_node", model.m_node.weight),
        ("m_edge", model.m_edge.weight),
        ("environment", next(model.environment.parameters())),
        ("pair", next(model.pair.parameters())),
        ("reader", next(model.reader.parameters())),
    ]:
        grad = parameter.grad
        assert grad is not None and torch.isfinite(grad).all(), name
        value = float(grad.abs().sum())
        assert value > 0.0, f"{name} gradient is exactly zero"
        report[name] = value
    return {"gradients": report, "verdict": "PASS"}


def check_g8_parameter_matching() -> dict[str, Any]:
    d_node, d_edge = _dictionary(seed=0)
    baseline = pec.build_model("dense", d_node=d_node, d_edge=d_edge, seed=0)
    target = pec.n_params(baseline)
    reader_hidden = match_reader_hidden(target)
    candidate = build_i1_model("dense", d_node=d_node, d_edge=d_edge, reader_hidden=reader_hidden, seed=0)
    candidate_params = pec.n_params(candidate)
    relative = abs(candidate_params - target) / target
    assert relative <= 0.01, (target, candidate_params, relative)
    assert reader_hidden == I1_READER_HIDDEN, reader_hidden
    return {
        "baseline_total_params": int(target),
        "candidate_total_params": int(candidate_params),
        "reader_hidden": int(reader_hidden),
        "relative_diff": float(relative),
        "verdict": "PASS",
    }


I1_GATES = {
    "G0_environment_equivalence": check_g0_environment_equivalence,
    "G1_pair_equivalence": check_g1_pair_equivalence,
    "G2_no_message_passing": check_g2_no_message_passing,
    "G3_environment_freeze": check_g3_environment_freeze,
    "G4_chemistry_purity": check_g4_chemistry_purity,
    "G5_relabel_invariance": check_g5_relabel_invariance,
    "G6_bucket_correctness": check_g6_bucket_correctness,
    "G7_gradients": check_g7_gradients,
    "G8_parameter_matching": check_g8_parameter_matching,
}


def run_i1_gates() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, function in I1_GATES.items():
        report[name] = function()
    return report
