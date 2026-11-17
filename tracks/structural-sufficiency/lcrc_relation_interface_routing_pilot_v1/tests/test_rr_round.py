"""Unit tests for round lcrc-relation-interface-routing-pilot-v1.

Fast: no trajectory is trained.  The tests assert the invariants that make the
formal runs interpretable — the frozen partition, the routing conservation laws,
the parameter budget, the shared initial weights, the binary block gate, the
label/oracle blindness of the selection score, and the declared protocol rules.
The expensive whole-split gates live in ``rr_checks`` and are run by the
``checks`` stage; here we only re-run the cheap ones on a small sample.
"""

from __future__ import annotations

import numpy as np
import pytest

import rr_common as C
import rr_partition as P


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_identity_constants() -> None:
    assert C.ROUND_ID == "lcrc-relation-interface-routing-pilot-v1"
    assert C.PROTOCOL_ID == C.ROUND_ID
    assert C.STUDY_ID == C.ROUND_ID
    assert C.TRAIN_SEED == 0
    assert C.TASKS == ("lipo", "wc_logp")
    assert C.MODELS == ("R1", "R2")
    assert C.DATASET == "ogbg-mollipo"
    assert C.EXPECTED_SPLIT_SIZES == {"train": 3360, "valid": 420, "test": 420}
    assert C.WIDTH == 64
    assert len({C.INIT_STREAM_BLOCK, C.INIT_STREAM_IFACE, C.INIT_STREAM_RHO, C.INIT_STREAM_CONTEXT}) == 4


def test_paths_point_into_this_round() -> None:
    assert C.CODE_DIR == C.ROUND_DIR / "code"
    assert C.RESULTS_DIR == C.ROUND_DIR / "results"
    assert C.ANALYSIS_DIR == C.ROUND_DIR / "analysis"
    for path in (C.SQL_CODE, C.RC_CODE, C.PREP_CODE, C.DISSECT_CODE):
        assert path.is_dir(), path
    for path in (C.SQL_DIR, C.RC_DIR, C.PREP_DIR, C.DISSECT_DIR, C.ROUTE_DIR):
        assert C.ROUND_DIR not in path.parents


def test_frozen_command_inputs_are_untouched() -> None:
    # the read-only reuse roots are inputs; this round must never write there
    for path in (C.ORACLE_CONTRIB_NPZ, C.WC_LOG_PRED_CSV, C.C1_LIPO_TRAJECTORY):
        assert path.is_file(), path


# ---------------------------------------------------------------------------
# partition rule
# ---------------------------------------------------------------------------


def _edges(*pairs):
    return [tuple(sorted(p)) for p in pairs]


def _part(n_atoms: int, edges, atom_type=None, bond_type=None):
    """Partition a toy molecule with default (uniform, label-free) type ids."""
    atom_type = [0] * n_atoms if atom_type is None else atom_type
    bond_type = [0] * len(edges) if bond_type is None else bond_type
    return P.partition_molecule(n_atoms, edges, atom_type, bond_type)


def test_partition_of_a_ring_is_one_cycle_block() -> None:
    edges = _edges((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0))
    part = _part(6, edges)
    check = P.check_partition(part, edges)
    assert check["bond_unique_ownership"]
    assert check["blocks_connected"]
    assert check["weights_sum_to_one"]
    assert part.n_blocks == 1
    assert [b["kind"] for b in part.blocks] == ["cycle"]


def test_partition_of_a_chain_splits_into_one_chain_block() -> None:
    edges = _edges((0, 1), (1, 2), (2, 3))
    part = _part(4, edges)
    assert part.n_blocks == 1
    assert [b["kind"] for b in part.blocks] == ["chain"]
    assert sorted(part.bond_block.tolist()) == [0, 0, 0]
    P.check_partition(part, edges)


def test_partition_separates_ring_from_its_substituent_chain() -> None:
    # benzene with a two-bond substituent: the ring bonds are non-bridges, the
    # substituent bonds are bridges, so they cannot share a block.
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    edges = _edges(*ring, (0, 6), (6, 7))
    part = _part(8, edges)
    kinds = sorted(b["kind"] for b in part.blocks)
    assert kinds == ["chain", "cycle"]
    owners = {i: int(part.bond_block[i]) for i in range(len(edges))}
    ring_block = owners[0]
    chain_block = owners[6]
    assert ring_block != chain_block
    # the shared atom 0 belongs to both blocks, with weights summing to 1
    weights = part.weights()
    assert len(weights[0]) == 2
    assert abs(sum(w for _, w in weights[0]) - 1.0) < 1e-12
    # atom 0 has degree 3: 2 ring bonds + 1 bridge -> weight 2/3 on the ring
    assert abs(dict(weights[0])[ring_block] - 2.0 / 3.0) < 1e-12


def test_partition_owns_every_bond_exactly_once() -> None:
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    edges = _edges(*ring, (0, 6), (6, 7), (7, 8), (8, 9), (9, 6))
    part = _part(10, edges)
    owned = np.zeros(len(edges), dtype=np.int64)
    for block in part.blocks:
        owned[list(block["bonds"])] += 1
    assert (owned == 1).all()
    P.check_partition(part, edges)


def test_isolated_atom_is_its_own_block() -> None:
    part = _part(2, [])
    assert part.n_blocks == 2
    assert all(b["kind"] == "isolated_atom" for b in part.blocks)
    weights = part.weights()
    assert len(weights) == 2
    assert all(abs(sum(w for _, w in row) - 1.0) < 1e-12 for row in weights)


def test_block_order_is_permutation_invariant() -> None:
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    edges = _edges(*ring, (0, 6), (6, 7))
    part = _part(8, edges)
    perm = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    permuted = _edges(*[(int(perm[u]), int(perm[v])) for u, v in edges])
    part2 = _part(8, permuted)
    assert part.n_blocks == part2.n_blocks
    assert sorted(b["kind"] for b in part.blocks) == sorted(b["kind"] for b in part2.blocks)
    assert sorted(len(b["bonds"]) for b in part.blocks) == sorted(len(b["bonds"]) for b in part2.blocks)


def test_bridge_mask_matches_bruteforce() -> None:
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    edges = _edges(*ring, (0, 6))
    mask = P.bridge_mask(7, edges)
    assert list(mask) == [False] * 6 + [True]


# ---------------------------------------------------------------------------
# routing conservation (cached, whole Valid split)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle():
    import rr_data as D

    return D.build_bundle()


def test_routing_cache_is_the_frozen_one(bundle) -> None:
    support = C.read_json(C.ANALYSIS_DIR / "partition_support.json")
    for split in C.SPLITS:
        assert support["routing_cache_file_sha256"][split] == C.sha256_file(C.routing_cache_path(split))


def test_s2_conservation_on_valid(bundle) -> None:
    stats = C.read_json(C.CACHE_DIR / "manifest.json")["splits"]["valid"]["stats"]
    assert int(stats["n_s2_internal"]) + int(stats["n_s2_cross"]) == int(stats["n_s2_total"])
    assert int(stats["n_s1_occurrences"]) == 2 * int(stats["n_bonds"])
    assert bool(stats["s2_internal_plus_cross_equals_total"])
    # every cross-boundary occurrence is owned by exactly one interface
    assert int(stats["n_s2_cross"]) > 0
    routing = bundle.routing["valid"]
    assert int(routing["path_cross"].sum()) == int(stats["n_s2_cross"])
    assert int((routing["path_iface"] >= 0).sum()) == int(stats["n_s2_cross"])
    # conservation: one destination per occurrence, internal XOR cross
    assert int((routing["path_cross"] == 0).sum()) == int(stats["n_s2_internal"])
    assert bool(((routing["path_bucket"] >= 0) == (routing["path_cross"] == 0)).all())
    assert bool(((routing["path_iface"] >= 0) == (routing["path_cross"] == 1)).all())
    assert int(np.diff(routing["iface_ptr"]).sum()) == int(stats["n_interfaces"])
    assert int(routing["nbr_bucket"].size) == int(stats["n_s1_occurrences"])


def test_partition_gate_passed_on_valid() -> None:
    support = C.read_json(C.ANALYSIS_DIR / "partition_support.json")
    assert support["gate"]["status"] == "PARTITION_USABLE"
    assert support["gate"]["pass"] is True
    assert support["gate"]["observed_fraction"] >= C.PARTITION_GATE_FRACTION
    assert support["gate"]["split"] == "valid"


# ---------------------------------------------------------------------------
# architecture and budget
# ---------------------------------------------------------------------------


def test_parameter_budget(bundle) -> None:
    import rr_models as M

    budget = C.read_json(C.ANALYSIS_DIR / "parameter_budget.json")
    r1, r2 = budget["models"]["R1"]["total"], budget["models"]["R2"]["total"]
    c1 = budget["c1_reference_total"]
    assert c1 == 228674
    assert abs(r1 / c1 - 1) <= 0.25
    assert abs(r2 / c1 - 1) <= 0.25
    assert abs(r2 - r1) / r1 <= C.CAPACITY_CONFOUNDED_RATIO
    assert budget["ratio"]["capacity_confounded"] is False
    # the analytic formulas must match the real modules
    counts = M.expected_parameter_counts(int(bundle.spec.global_dim))
    assert counts["R1_total"] == r1 and counts["R2_total"] == r2


def test_models_share_the_frozen_trunk_init(bundle) -> None:
    import rr_models as M

    canonical = M.canonical_trunk_state(bundle.spec, bundle.norm)
    model1 = M.build_model("R1", bundle.spec, bundle.norm, canonical, device="cpu")
    model2 = M.build_model("R2", bundle.spec, bundle.norm, canonical, device="cpu")
    s1, s2 = model1.state_dict(), model2.state_dict()
    shared = [k for k in s1 if k in s2 and "context" not in k]
    assert len(shared) > 40
    for key in shared:
        assert (s1[key] == s2[key]).all(), key


def test_binary_block_gate_matches_forward(bundle) -> None:
    import rr_checks as K

    out = K.gates_ledger_and_gate(bundle, max_molecules=8)
    assert out["pass"] is True


# ---------------------------------------------------------------------------
# protocol / configuration declarations
# ---------------------------------------------------------------------------


def test_selection_score_is_frozen_and_label_free() -> None:
    import ast
    import inspect

    import rr_checks as K

    params = set(inspect.signature(K.select_blocks).parameters)
    assert not (params & {"y", "targets", "labels", "oracle", "wc_logp", "rows"})
    tree = ast.parse(inspect.getsource(K.select_blocks))
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    forbidden = {"oracle", "wc_logp", "y_raw", "y", "targets", "labels", "target", "edit"}
    assert not (identifiers & forbidden), identifiers & forbidden
    # the score is |delta_i| / sqrt(max(1, n_bonds_i)) and ties keep the smallest index
    source = inspect.getsource(K.select_blocks)
    assert "np.sqrt" in source and "argmax" in source


def test_recipe_is_frozen() -> None:
    sig = __import__("inspect").signature
    assert C.RECIPE["epochs"] == 360
    assert C.RECIPE["batch_size"] == 128
    assert C.RECIPE["peak_lr"] == pytest.approx(1e-3)
    assert C.RECIPE["min_lr"] == pytest.approx(1e-5)
    assert C.RECIPE["warmup_epochs"] == 5
    assert C.RECIPE["weight_decay"] == pytest.approx(1e-5)
    assert C.RECIPE["dropout"] == pytest.approx(0.10)
    assert C.RECIPE["gradient_clip"] == pytest.approx(5.0)
    assert C.RECIPE["amp"] is False
    assert C.RECIPE["early_stopping"] is False
    assert C.RECIPE["soup"] is False


def test_lr_schedule_matches_the_shared_implementation() -> None:
    import rr_train as T

    out = T.schedule_audit()
    assert out["max_abs_diff"] == 0.0
    assert out["n_steps"] == T.total_steps(3360) == 360 * 27


def test_protocol_declares_test_policy_always() -> None:
    import yaml

    protocol = yaml.safe_load(C.ROUND_DIR.joinpath("protocols", f"{C.PROTOCOL_ID}.yaml").read_text())
    assert protocol["test_policy"] == "always"
    assert protocol["validation_selection_policy"]["forbidden"].startswith("selecting an epoch by Test")
    assert protocol["split"]["sizes"]["ogbg-mollipo"] == {"train": 3360, "valid": 420, "test": 420}
    assert protocol["seed_semantics"]["seeds"] == [0]
    assert protocol["comparability"]["ranking_metric"] == "valid_rmse"


def test_study_and_config_parse() -> None:
    import yaml

    study = yaml.safe_load(C.ROUND_DIR.joinpath("studies", f"{C.STUDY_ID}.yaml").read_text())
    assert study["id"] == C.STUDY_ID
    assert [c["id"] for c in study["candidates"]] == ["R1", "R2"]
    config = yaml.safe_load(C.FROZEN_CONFIG.read_text())
    assert config["protocol_id"] == C.PROTOCOL_ID
    assert config["seed"] == 0
    assert config["model"] in C.MODELS
