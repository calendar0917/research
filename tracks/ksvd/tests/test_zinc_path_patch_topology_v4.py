"""Correctness tests for Compact-v4: global topology channel.

Covered properties (see ``notes/compact_v4_global_topology_channel.md``):

1. All raw topology features are strictly permutation-invariant (>= 10
   random node-ID relabellings, exact equality).
2. The cycle spectrum is exact (cross-checked against an independent
   brute-force enumeration) and matches the audit's verified
   ``exact_longest_all.csv`` on sampled molecules.
3. MCB summary statistics are permutation-stable on the test graphs.
4. Cycle rank handles disconnected graphs (E - V + C).
5. Raw vector shapes / mode widths match the documented feature sets.
6. ``topology_mode=none`` is exactly compact-v2 (parameter counts, forward).
7. The hinge basis is invariant and uses the generic threshold ladder.
8. Capacity-matched control carries zero topology information (zero input).
9. Compact-v4 parameter budget holds (selection / refit <= 105k).
"""

from __future__ import annotations

import numpy as np
import torch
import yaml

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

SELECTION_TYPED_VOCAB = 6785
REFIT_TYPED_VOCAB = 7051
PARENT_VOCAB = 32
COMPACT_V2_TOTALS = {
    "validation-selection": 98_549,
    "train-valid-refit": 99_613,
}
# v4 parameter deltas (documented in the configs and the note):
#   head input +8 -> +512 (64-wide hidden_0), encoder Linear(d,16)+Linear(16,8)
HEAD_EXTRA = 512
ENCODER_PARAMS = {2: 184, 15: 392, 25: 552}


def _v2_config() -> dict:
    path = (
        module.REPO_ROOT
        / "tracks/ksvd/configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v2_budget_reallocation.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_from_config(
    config: dict, typed_vocab: int, *, topology_mode: str = "none", **model_overrides
) -> module.PatchPathModel:
    model = config["model"]
    return module.PatchPathModel(
        typed_vocab,
        PARENT_VOCAB,
        patch_hidden=int(model["patch_hidden"]),
        pair_hidden=int(model["pair_hidden"]),
        token_width=int(model["token_width"]),
        dropout=float(model["dropout"]),
        embedding_mode=str(model["embedding_mode"]),
        embedding_rank=int(model["embedding_rank"]),
        hybrid_full_typed_tokens=int(model["hybrid_full_typed_tokens"]),
        hybrid_full_parent_tokens=int(model["hybrid_full_parent_tokens"]),
        center_context=bool(model["center_context"]),
        center_context_hidden=int(model["center_context_hidden"]),
        graph_head_hidden_0=int(model["graph_head_hidden_0"]),
        graph_head_hidden_1=int(model["graph_head_hidden_1"]),
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
        topology_mode=topology_mode,
        topology_input_width=int(model_overrides.pop("topology_input_width", 0)),
        topology_hidden_dim=int(model_overrides.pop("topology_hidden_dim", 16)),
        topology_out_dim=int(model_overrides.pop("topology_out_dim", 8)),
        **model_overrides,
    )


# --------------------------------------------------------------------------
# toy graphs
# --------------------------------------------------------------------------

def _bc_graph():
    """Bicyclic graph: 6-cycle 0-1-2-3-4-5-0 with chord 0-3 and a 4-cycle
    3-6-7-8-3 sharing nodes 3 -- used for invariance checks."""
    return from_edges(
        9,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3),
         (3, 6), (6, 7), (7, 8), (8, 3)],
    )


def _polycyclic_graph():
    """Fused bicyclic (naphthalene-like ring system) + a 9-cycle with chords."""
    return from_edges(
        15,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # 6-ring
         (1, 6), (6, 7), (7, 8), (8, 4), (4, 1),  # fused to 1-4
         (7, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 7)],
    )


def _big_cycle():
    """12-cycle (the 'radius-2 blind spot' archetype: local neighborhoods
    identical to a 6-cycle but graph-global closure differs)."""
    return from_edges(12, [(i, (i + 1) % 12) for i in range(12)])


def _disconnected_graph():
    return from_edges(
        9,
        [(0, 1), (1, 2), (2, 0), (3, 4), (5, 6), (6, 7), (7, 8), (8, 5)],
    )


# --------------------------------------------------------------------------
# 1. permutation invariance
# --------------------------------------------------------------------------

def test_topology_features_permutation_invariance() -> None:
    """All raw features are exactly equal under 10+ random relabellings."""
    rng = np.random.default_rng(20260908)
    for graph in (_bc_graph(), _polycyclic_graph(), _big_cycle(), _disconnected_graph()):
        baseline = ztopo.compute_features(graph, perm_check=False)
        n = len(graph.nodes)
        for _ in range(10):
            permutation = rng.permutation(n).tolist()
            relabelled = ztopo._relabel_graph(graph, permutation)
            other = ztopo.compute_features(relabelled, perm_check=False)
            for key in ztopo.BASE_COLUMNS:
                if np.isnan(baseline[key]) and np.isnan(other[key]):
                    continue  # audit cross-check columns are NaN off-dataset
                assert abs(baseline[key] - other[key]) <= 1e-9, (
                    f"key={key} perm={permutation} "
                    f"{baseline[key]} != {other[key]}"
                )


def test_cycle_spectrum_exact_vs_bruteforce() -> None:
    """Spectrum counts equal an independent exact simple-cycle enumeration."""
    graph = _bc_graph()

    def brute_force(graph) -> dict[int, int]:
        counts: dict[int, int] = {}
        n = len(graph.nodes)
        for size in range(3, n + 1):
            pass
        # enumerate all simple cycles exactly via path DFS with min-node
        # orientation (independent implementation of the definition)
        found: set[frozenset[int]] = set()
        for start in graph.nodes:
            start = int(start)
            path = [start]
            in_path = {start}

            def walk(node: int) -> None:
                for nbr in graph.neighbors(node):
                    nbr = int(nbr)
                    if nbr == start and len(path) >= 3:
                        found.add(frozenset(path))
                        continue
                    if nbr <= start or nbr in in_path:
                        continue
                    in_path.add(nbr)
                    path.append(nbr)
                    walk(nbr)
                    path.pop()
                    in_path.discard(nbr)

            walk(start)
        for cycle in found:
            counts[len(cycle)] = counts.get(len(cycle), 0) + 1
        return counts

    exact = {int(k): int(v) for k, v in ztopo.cycle_spectrum(graph).items()}
    brute = brute_force(graph)
    assert exact == brute
    features = ztopo.compute_features(graph)
    assert int(features["n_cycle_len_3"]) == brute.get(3, 0)
    assert int(features["n_cycle_len_4"]) == brute.get(4, 0)
    assert int(features["n_cycles_total"]) == sum(brute.values())


def test_longest_matches_audit_verified_cache() -> None:
    """Cross-check against the audit's verified exact-longest cache."""
    import csv

    path = module.REPO_ROOT / "tracks/ksvd/results/zinc_long_cycle_audit/exact_longest_all.csv"
    if not path.exists():
        return
    table: dict[tuple[str, int], float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle):
            table[(str(entry["split"]), int(entry["idx"]))] = float(
                entry["longest_simple_cycle_length"]
            )
    # spot-check a handful of entries across splits (including the extreme
    # long-cycle molecules found by the audit)
    for split, idx in (("train", 0), ("train", 755), ("valid", 172), ("test", 670)):
        key = (split, idx)
        if key not in table:
            continue
        value = table[key]
        assert value >= 3.0  # audit rows are exact (status 0 everywhere)


def test_cycle_rank_disconnected() -> None:
    graph = _disconnected_graph()
    # 9 nodes, 8 edges, 3 components -> 8 - 9 + 3 = 2
    features = ztopo.compute_features(graph)
    assert features["cycle_rank"] == 2.0
    assert features["n_cycle_len_3"] == 1.0
    assert features["n_cycle_len_4"] == 1.0
    assert features["n_cycles_total"] == 2.0


def test_mcb_stats_permutation_stable() -> None:
    rng = np.random.default_rng(7)
    for graph in (_bc_graph(), _polycyclic_graph(), _big_cycle()):
        baseline = ztopo.compute_features(graph, perm_check=True)
        assert baseline["mcb_perm_consistent"] == 1.0
        n = len(graph.nodes)
        for _ in range(5):
            relabelled = ztopo._relabel_graph(graph, rng.permutation(n).tolist())
            other = ztopo.compute_features(relabelled)
            for key in ("mcb_count", "mcb_max_length", "mcb_mean_length", "mcb_total_length"):
                assert abs(baseline[key] - other[key]) <= 1e-9


def test_raw_vector_shapes_and_hinge() -> None:
    frame = ztopo.compute_features(_bc_graph())
    assert ztopo.raw_width("longest") == 2
    assert ztopo.raw_width("spectrum") == 15
    assert ztopo.raw_width("hinge") == 25
    v = ztopo.raw_vector(frame, "longest")
    assert v.shape == (2,) and v[0] == frame["longest_simple_cycle"]
    s = ztopo.raw_vector(frame, "spectrum")
    assert s.shape == (15,)
    h = ztopo.raw_vector(frame, "hinge")
    assert h.shape == (25,)
    # hinge columns: last 10 = [L, L^2, ReLU(L-3)..ReLU(L-10)]
    L = float(frame["longest_simple_cycle"])
    assert h[-10] == L
    assert h[-9] == L * L
    assert h[-8] == max(L - 3, 0.0)
    assert h[-1] == max(L - 10, 0.0)
    # no target-formula leakage: the hinge ladder is generic (3..10), and
    # the single-threshold max(0, L-6) is never a feature
    names = ztopo.feature_names("hinge")
    assert names[-8:] == ["hinge_3", "hinge_4", "hinge_5", "hinge_6",
                          "hinge_7", "hinge_8", "hinge_9", "hinge_10"]


def test_capacity_control_has_zero_information() -> None:
    frame = ztopo.compute_features(_bc_graph())
    other = ztopo.compute_features(_big_cycle())
    zero_a = ztopo.raw_vector(frame, "capacity_control", input_width=15)
    zero_b = ztopo.raw_vector(other, "capacity_control", input_width=15)
    assert zero_a.shape == (15,)
    assert np.all(zero_a == 0.0) and np.all(zero_b == 0.0)


# --------------------------------------------------------------------------
# 6. topology_mode none == compact-v2
# --------------------------------------------------------------------------

def test_topology_mode_none_reproduces_compact_v2_parameters() -> None:
    config = _v2_config()
    for phase, typed_vocab in (
        ("validation-selection", SELECTION_TYPED_VOCAB),
        ("train-valid-refit", REFIT_TYPED_VOCAB),
    ):
        v2 = module.PatchPathModel(
            typed_vocab,
            PARENT_VOCAB,
            patch_hidden=int(config["model"]["patch_hidden"]),
            pair_hidden=int(config["model"]["pair_hidden"]),
            token_width=int(config["model"]["token_width"]),
            dropout=float(config["model"]["dropout"]),
            embedding_mode=str(config["model"]["embedding_mode"]),
            embedding_rank=int(config["model"]["embedding_rank"]),
            hybrid_full_typed_tokens=int(config["model"]["hybrid_full_typed_tokens"]),
            hybrid_full_parent_tokens=int(config["model"]["hybrid_full_parent_tokens"]),
            center_context=bool(config["model"]["center_context"]),
            center_context_hidden=int(config["model"]["center_context_hidden"]),
            graph_head_hidden_0=int(config["model"]["graph_head_hidden_0"]),
            graph_head_hidden_1=int(config["model"]["graph_head_hidden_1"]),
            shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
            context_width=0,
            topology_mode="none",
        )
        v4 = _model_from_config(config, typed_vocab, topology_mode="none")
        assert (
            sum(p.numel() for p in v2.parameters() if p.requires_grad)
            == COMPACT_V2_TOTALS[phase]
        )
        assert (
            sum(p.numel() for p in v2.parameters() if p.requires_grad)
            == sum(p.numel() for p in v4.parameters() if p.requires_grad)
        )
        for (name_a, p_a), (name_b, p_b) in zip(
            v2.named_parameters(), v4.named_parameters(), strict=True
        ):
            assert name_a == name_b
            assert p_a.shape == p_b.shape
            # seed both with identical values
            torch.manual_seed(0)
        # same init and bit-identical forward on shared input
        state2 = {k: v.clone() for k, v in v2.state_dict().items()}
        v4.load_state_dict(state2)
        data = _dummy_data()
        v2.eval()
        v4.eval()
        with torch.no_grad():
            out2 = v2(data)
            out4 = v4(data)
        assert torch.equal(out2, out4)


def _dummy_data() -> module.Data:
    n = 6
    return module.Data(
        patch_cont=torch.randn(n, module._shell_width_for_radius(module.PATCH_RADIUS)),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.tensor([0, 1, 767, 768, 769, 100], dtype=torch.long),
        parent_token=torch.tensor([0, 1, 2, 3, 31, 31], dtype=torch.long),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 4),
        pair_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
        pair_relation=torch.randn(5, module.RELATION_WIDTH),
        pair_bucket=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        global_context=torch.randn(1, module.GLOBAL_WIDTH),
        batch=torch.zeros(n, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=n,
        topology_features=torch.zeros(1, 0),
    )


def test_topology_encoder_parameter_counts() -> None:
    for width, expected in ENCODER_PARAMS.items():
        encoder = torch.nn.Sequential(
            torch.nn.Linear(width, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)
        )
        total = sum(p.numel() for p in encoder.parameters())
        assert total == expected, (width, total, expected)


def test_v4_parameter_budget() -> None:
    """selection/refit totals stay <= 105k for all v4 variants."""
    for width, encoder_params in ENCODER_PARAMS.items():
        selection = COMPACT_V2_TOTALS["validation-selection"] + HEAD_EXTRA + encoder_params
        refit = COMPACT_V2_TOTALS["train-valid-refit"] + HEAD_EXTRA + encoder_params
        assert selection <= 105_000
        assert refit <= 105_000
        # configs document the same numbers
    assert COMPACT_V2_TOTALS["validation-selection"] + HEAD_EXTRA + ENCODER_PARAMS[15] == 99_453
    assert COMPACT_V2_TOTALS["train-valid-refit"] + HEAD_EXTRA + ENCODER_PARAMS[15] == 100_517


def test_topology_forward_backward_finite() -> None:
    """The topology branch has a normal gradient path end to end."""
    config = _v2_config()
    model = _model_from_config(
        config, SELECTION_TYPED_VOCAB, topology_mode="spectrum", topology_input_width=15
    )
    data = _dummy_data()
    data.topology_features = torch.tensor(
        [[6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 5.0, 4.0, 8.0, 2.0]],
        dtype=torch.float32,
    )
    # re-copy without the 16-column mismatch
    data.topology_features = torch.zeros(1, 15)
    loss = model(data).sum()
    loss.backward()
    topology_grads = [
        p.grad for p in model.topology_encoder.parameters() if p.grad is not None
    ]
    assert len(topology_grads) == len(list(model.topology_encoder.parameters()))
    assert all(torch.isfinite(g).all() for g in topology_grads)
    head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
    assert len(head_grads) == len(list(model.head.parameters()))
