"""cscl-distance-pooling-bottleneck-v1 correctness tests (fast, CPU).

These tests pin the frozen Soft5 kernel and pooling arithmetic, the Hard5
bit-identity to the historical path, source-level discipline guards (seed 0
only, no official test, no label access before freeze, no kernel search) and
the pre-registered decision bands.  Heavy data gates (10k distance census,
purchase gate, GPU arms) live in the runner stages and in the results
directory.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
for p in (str(_CODE_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/distance_pooling_bottleneck_v1.yaml"
HIST_PREP = (
    REPO_ROOT
    / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1/all_train_prep.npz"
)
HIST_Y_SOUP = (
    REPO_ROOT
    / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1/Y_seed0_raw_soup_state.pt"
)


def _dp():
    import distance_pooling_v1 as dp

    return dp


def _driver():
    import run_distance_pooling_v1 as drv

    return drv


def _driver_source() -> str:
    return Path(_driver().__file__).read_text(encoding="utf-8")


def _strip_docstrings(text: str) -> str:
    class _Strip(ast.NodeTransformer):
        def _drop(self, node):
            self.generic_visit(node)
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
            return node

        visit_Module = _drop
        visit_FunctionDef = _drop
        visit_AsyncFunctionDef = _drop
        visit_ClassDef = _drop

    return ast.unparse(_Strip().visit(ast.parse(text)))


# ---------------------------------------------------------------------------
# frozen protocol constants
# ---------------------------------------------------------------------------


def test_frozen_soft5_kernel():
    dp = _dp()
    assert dp.PROTOCOL_ID == "cscl-distance-pooling-bottleneck-v1"
    assert dp.SOFT5_CENTERS == (1, 2, 3, 4, 8)
    assert dp.SOFT5_SIGMA == 0.35
    assert dp.N_BUCKETS == 5
    assert dp.FULL_PARAMETERS == 408_651
    assert dp.READER_INPUT_DIM == 814
    assert dp.RELATION_READOUT_DIM == 485
    assert dp.PAIR_VALUE_DIM == 48
    assert dp.PAIR_POOLING_MODES == ("hard5", "soft5")


def test_frozen_driver_constants():
    drv = _driver()
    assert drv.EPOCHS == 240
    assert drv.LR == 1.0e-3
    assert drv.WEIGHT_DECAY == 1.0e-5
    assert drv.GRAD_CLIP == 5.0
    assert drv.BATCH_SIZE == 128
    assert drv.SOUP_LAST == 5
    assert drv.TRAIN_SHUFFLE_OFFSET == 101
    assert drv.FULL_PARAMETERS == 408_651
    assert abs(drv.H1_LAMBDA - 33.95873017865987) < 1e-12
    assert drv.EXPECTED_STEPS == 18_960
    assert drv.BOOTSTRAP_N == 5000
    assert drv.BOOTSTRAP_SEED == 0
    assert drv.DECISION_DELTA_A == -0.010
    assert drv.DECISION_DELTA_B == -0.003
    assert drv.ARM_NAMES == ("Hard5", "Soft5")
    assert set(drv.ARM_MODES.values()) == {"hard5", "soft5"}


def test_config_matches_driver():
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == "cscl-distance-pooling-bottleneck-v1"
    assert cfg["soft5_operator"]["centers"] == [1, 2, 3, 4, 8]
    assert cfg["soft5_operator"]["sigma"] == 0.35
    assert cfg["soft5_operator"]["parameters_added"] == 0
    assert cfg["soft5_operator"]["explicit_non_variants"]
    assert cfg["source_path_audit"]["parameters_expected"] == 408_651
    assert cfg["source_path_audit"]["reader_input_expected"].endswith("= 814")
    assert cfg["training"]["steps_expected"] == 18_960
    assert cfg["training"]["soup"].startswith("parameter mean of epochs 236..240")
    assert cfg["training"]["train_shuffle_seed"] == 101
    assert cfg["purchase_gate"]["G1_long_distance_heterogeneity"]["threshold"] == ">= 0.10"
    assert cfg["purchase_gate"]["G2_feature_difference"]["threshold"] == ">= 0.25"
    assert cfg["purchase_gate"]["on_failure"].startswith("CPU_NO_PURCHASE")
    assert cfg["decision_rule"]["case_A"].split("->")[-1].strip() == "SOFT5_PREDICTIVE_SIGNAL"
    assert cfg["decision_rule"]["case_B"].startswith("-0.010 < delta <= -0.003")
    assert cfg["decision_rule"]["case_C"].startswith("delta > -0.003")
    assert cfg["discipline"]["seeds"] == [0]
    assert cfg["discipline"]["no_official_test"] is True
    assert cfg["discipline"]["no_bucket_center_width_search"] is True
    assert cfg["discipline"]["no_new_trainable_parameters"] is True
    assert cfg["evaluation"]["official_test"] == "forbidden"
    assert cfg["budget"]["formal_new_training_trajectories"] == 2


# ---------------------------------------------------------------------------
# Soft5 kernel mathematics
# ---------------------------------------------------------------------------


def test_soft5_weights_are_a_normalised_nonnegative_basis():
    dp = _dp()
    log_distance = torch.log1p(torch.arange(1, 23, dtype=torch.float32))
    weights = dp.soft5_weights_from_log_distance(log_distance)
    assert tuple(weights.shape) == (22, 5)
    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0).all())
    assert float((weights.sum(1) - 1.0).abs().max()) <= 1e-6
    # every basis has a non-trivial active range (not a degenerate one-hot)
    assert float(weights.std(0).min()) > 1e-3
    # the closest center dominates at each integer distance
    assert int(weights.argmax(1)[0]) == 0     # d=1 -> center 1
    assert int(weights.argmax(1)[7]) == 4     # d=8 -> center 8
    assert int(weights.argmax(1)[-1]) == 4    # d=22 -> center 8


def test_soft5_weights_differ_from_hard_one_hot_including_inside_bucket4():
    dp = _dp()
    distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 22.0])
    log_distance = torch.log1p(distances)
    soft = dp.soft5_weights_from_log_distance(log_distance)
    hard = dp.hard5_weights_from_bucket(torch.clip(distances.long(), 1, 5) - 1)
    # nothing in the last bucket is a near-identity copy of the one-hot
    for row in (4, 5, 6, 7):
        assert float((soft[row] - hard[row]).abs().max()) > 0.03
    # d=5 and d=8 must not receive the same soft mass pattern
    assert float((soft[4] - soft[6]).abs().max()) > 0.1
    # the last bucket receives real mass at its two extreme distances
    assert float(soft[4, 4]) > 0.1 and float(soft[8, 4]) > 0.9


def test_kernel_constants_are_literal_not_searched():
    """sigma/centers must be module literals with no data/y dependence."""
    fn = next(
        node for node in ast.walk(ast.parse(Path(_dp().__file__).read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "soft5_weights_from_log_distance"
    )
    src = ast.unparse(fn)
    assert "SOFT5_CENTERS" in src and "SOFT5_SIGMA" in src
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            assert node.id not in {"y", "fit", "valid", "self", "data"}, node.id
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"y", "fit", "valid", "data"}, node.attr
    for forbidden in ("search", "optuna", "nn.Parameter", "requires_grad"):
        assert forbidden not in src, forbidden


# ---------------------------------------------------------------------------
# pooling arithmetic
# ---------------------------------------------------------------------------


def _synthetic_pairs(seed: int = 0, n_graphs: int = 7, per_graph: int = 11):
    rng = torch.Generator().manual_seed(seed)
    pair_batch = torch.arange(n_graphs).repeat_interleave(per_graph).to(torch.long)
    value = torch.randn(n_graphs * per_graph, 48, generator=rng)
    distance = torch.randint(1, 16, (n_graphs * per_graph,), generator=rng).to(torch.float32)
    log_distance = torch.log1p(distance)
    pair_bucket = torch.clip(distance.to(torch.long), 1, 5) - 1
    return value, pair_batch, pair_bucket, log_distance, n_graphs


def test_hard5_pooling_is_bit_identical_to_the_frozen_path():
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit

    dp = _dp()
    value, pair_batch, pair_bucket, _, n_graphs = _synthetic_pairs()
    ours = dp.pool_pair_moments_hard5(value, pair_batch, pair_bucket, n_graphs, zero_count_blocks=True)
    frozen = audit.pool_pair_moments_masked(value, pair_batch, pair_bucket, n_graphs, ("count",), None)
    assert torch.equal(ours, frozen)


def test_soft5_first_and_second_moment_conservation():
    dp = _dp()
    value, pair_batch, pair_bucket, log_distance, n_graphs = _synthetic_pairs()
    weights = dp.soft5_weights_from_log_distance(log_distance)
    out = dp.pool_pair_moments_soft5(value, pair_batch, log_distance, n_graphs, zero_count_blocks=True)
    view = out.reshape(n_graphs, 5, 97)
    reference = value.to(torch.float64)
    assert float(
        (view[:, :, :48].sum(1).to(torch.float64) - torch.zeros((n_graphs, 48), dtype=torch.float64).index_add_(0, pair_batch, reference)).abs().max()
    ) <= 1e-4
    assert float(
        (view[:, :, 48:96].sum(1).to(torch.float64) - torch.zeros((n_graphs, 48), dtype=torch.float64).index_add_(0, pair_batch, reference * reference)).abs().max()
    ) <= 1e-4


def test_soft5_second_moment_is_weighted_sum_of_squares():
    """Second moment must be sum_i w_i h_i^2 (not (sum w h)^2 nor sum (w h)^2)."""
    dp = _dp()
    value = torch.zeros(2, 48)
    value[0, 0], value[0, 1], value[1, 0], value[1, 1] = 1.0, 2.0, 3.0, 4.0
    pair_batch = torch.tensor([0, 0])
    log_distance = torch.log1p(torch.tensor([3.0, 4.0]))
    out = dp.pool_pair_moments_soft5(value, pair_batch, log_distance, 1, zero_count_blocks=True)
    view = out.reshape(1, 5, 97)
    weights = dp.soft5_weights_from_log_distance(log_distance)  # [2, 5]
    assert float(weights[:, :4].min()) > 0.01  # genuinely mixed weights
    for k in range(5):
        expected = (weights[:, k : k + 1] * value * value).sum(0)
        assert torch.allclose(view[0, k, 48:96], expected, atol=1e-6)
        first = view[0, k, :48]
        assert not torch.allclose(view[0, k, 48:96], first * first, atol=1e-3)
        assert not torch.allclose(
            view[0, k, 48:96], ((weights[:, k : k + 1] * value) ** 2).sum(0), atol=1e-3
        )


def test_soft5_counts_are_weighted_once_and_zeroed_under_c6():
    dp = _dp()
    value, pair_batch, _, log_distance, n_graphs = _synthetic_pairs()
    weights = dp.soft5_weights_from_log_distance(log_distance)
    zeroed = dp.pool_pair_moments_soft5(value, pair_batch, log_distance, n_graphs, zero_count_blocks=True)
    counts = zeroed.reshape(n_graphs, 5, 97)[:, :, 96]
    assert torch.equal(counts, torch.zeros_like(counts))
    with_counts = dp.pool_pair_moments_soft5(value, pair_batch, log_distance, n_graphs, zero_count_blocks=False)
    raw = with_counts.reshape(n_graphs, 5, 97)[:, :, 96]
    expected = torch.zeros((n_graphs, 5))
    for k in range(5):
        expected[:, k].index_add_(0, pair_batch, weights[:, k])
    assert torch.allclose(raw, torch.log1p(expected), atol=1e-5)
    assert float((expected.sum(1) - torch.bincount(pair_batch, minlength=n_graphs).float()).abs().max()) <= 1e-4


def test_pooling_is_pair_order_invariant():
    dp = _dp()
    value, pair_batch, pair_bucket, log_distance, n_graphs = _synthetic_pairs(seed=3)
    order = torch.randperm(value.shape[0], generator=torch.Generator().manual_seed(7))
    a = dp.pool_pair_moments_soft5(value, pair_batch, log_distance, n_graphs, zero_count_blocks=True)
    b = dp.pool_pair_moments_soft5(
        value[order], pair_batch[order], log_distance[order], n_graphs, zero_count_blocks=True
    )
    assert float((a - b).abs().max()) <= 1e-5
    ha = dp.pool_pair_moments_hard5(value, pair_batch, pair_bucket, n_graphs, zero_count_blocks=True)
    hb = dp.pool_pair_moments_hard5(
        value[order], pair_batch[order], pair_bucket[order], n_graphs, zero_count_blocks=True
    )
    assert float((ha - hb).abs().max()) <= 1e-5


def test_moment_blocks_never_mix_across_buckets():
    """The 485 layout is bucket-major [first 48 | second 48 | count 1] x 5."""
    dp = _dp()
    value, pair_batch, _, log_distance, n_graphs = _synthetic_pairs(seed=5)
    out = dp.pool_pair_moments_soft5(value, pair_batch, log_distance, n_graphs, zero_count_blocks=True)
    assert tuple(out.shape) == (n_graphs, 485)
    view = out.reshape(n_graphs, 5, 97)
    weights = dp.soft5_weights_from_log_distance(log_distance).to(torch.float64)
    for k in range(5):
        expected = torch.zeros((n_graphs, 48), dtype=torch.float64).index_add_(
            0, pair_batch, value.to(torch.float64) * weights[:, k : k + 1]
        )
        assert torch.allclose(view[:, k, :48].to(torch.float64), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# discipline guards (source level)
# ---------------------------------------------------------------------------


def test_seed_zero_only():
    assert "choices=[0]" in _driver_source()


def test_no_official_test_access():
    code = _strip_docstrings(_driver_source())
    for forbidden in ('split="test"', "split='test'", "load_test_data", "encoded_test"):
        assert forbidden not in code, f"driver must never touch official test ({forbidden})"


def test_only_two_arms_no_kernel_search():
    src = _driver_source()
    assert 'ARM_NAMES = ("Hard5", "Soft5")' in src
    for forbidden in ("sigma=", "centers=", "Soft6", "Hard6", "seed=1"):
        assert forbidden.replace("=", "=") not in _strip_docstrings(src), forbidden


def test_predict_stage_does_not_read_labels_before_freeze():
    tree = ast.parse(_driver_source())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "stage_predict"
    )
    src = ast.unparse(fn)
    assert ".y" not in src
    assert "np.savez_compressed" in src
    assert "freeze_manifest.json" in src and "write_json" in src
    assert "extraction_program_sha256" in src


def test_score_hashes_before_reading_official_y():
    tree = ast.parse(_driver_source())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "stage_score"
    )
    src = ast.unparse(fn)
    i_hash = src.index("extraction_program_sha256")
    i_y = src.index("_load_official_valid_y()")
    assert i_hash < i_y, "freeze re-hash must precede the single y read"


def test_train_loss_is_raw_y_plus_label_free_reconstruction():
    tree = ast.parse(_driver_source())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_arm"
    )
    src = ast.unparse(fn)
    assert "torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1))" in src
    assert "model.reconstruction_loss(aux['phi'], aux['coord'])" in src
    assert "loss = task + H1_LAMBDA * rec" in src
    assert "mask=cm.C6_MASK" in src
    assert "torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)" in src


def test_forward_differs_from_frozen_path_only_in_the_pooling_call():
    import inspect

    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit

    dp = _dp()
    src = inspect.getsource(dp.DistancePoolingFullY.forward)
    # exactly one call per mode, dispatched on the plain attribute
    assert src.count("pool_pair_moments_soft5(") == 1
    assert src.count("pool_pair_moments_hard5(") == 1
    assert src.count("audit.pool_pair_moments_masked") == 0
    assert 'if self.pair_pooling == "soft5":' in src
    # the frozen pipeline steps are all present, in the AuditModel order
    frozen_steps = [
        "self.code(data.dict_phi)",
        "self.environments_masked(coord, data, mask, fill)",
        "audit.pool_moments_masked(E, batch, n_graphs, mask.unary_zero_blocks, fill)",
        "self.pair_projection(E)",
        "data.pair_relation[:, list(audit.p1.P1_RELATION_INDICES)]",
        "self.relation_encoder(relation_input)",
        "1.0 + torch.tanh(self.distance_gate(data.pair_bucket))",
        "self.pair_encoder(pair_input)",
        "self.global_encoder(global_input)",
        "self.topology_encoder(topology_input)",
        "torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)",
        "self.reader(unified)",
    ]
    positions = [src.index(step) for step in frozen_steps]
    assert positions == sorted(positions), "frozen forward steps reordered"
    assert src.count("if mask is None") == 1 and "super().forward" in src
    # the frozen audit module forward itself is untouched by this round
    assert "pool_pair_moments_soft5" not in inspect.getsource(audit.AuditModel.forward)


def test_soft5_weights_read_the_already_consumed_log_distance_column():
    dp = _dp()
    src = Path(_dp().__file__).read_text(encoding="utf-8")
    assert "data.pair_relation[:, 5]" in src
    assert "SOFT5_CENTERS" in src and "SOFT5_SIGMA" in src
    # no label access anywhere in the operator module
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Attribute):
            assert node.attr != "y", "distance_pooling_v1 must not read labels"
        if isinstance(node, ast.Name):
            assert node.id not in {"decomp", "logp", "sa"}, node.id


# ---------------------------------------------------------------------------
# model construction (local data artifacts; skipped when absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HIST_PREP.exists(), reason="prep artifact not present")
def test_historical_reference_checkpoint_is_pinned():
    """The audit's read-only Hard5 reference must be the frozen confirmation-dir soup."""
    drv = _driver()
    assert drv.HIST_Y_SOUP.parent.name == "zinc_full_decomposition_valid_test_confirmation_v1"
    assert drv.HIST_Y_SOUP_SHA256 == (
        "7792c56045459a37c569424501352afb4f3fd8ebc06643522192710bfe85ecfe"
    )
    if HIST_Y_SOUP.exists():
        assert drv.file_sha256(HIST_Y_SOUP) == drv.HIST_Y_SOUP_SHA256


def test_matching_audit_uses_training_program_hash():
    src = _driver_source()
    assert "training_program_sha256" in src
    assert "same_training_program_sha256" in src


@pytest.mark.skipif(not HIST_PREP.exists(), reason="prep artifact not present")
def test_both_arms_share_the_frozen_initialisation_and_parameter_count():
    dp = _dp()
    prep = {k: z for k, z in np.load(HIST_PREP, allow_pickle=False).items()}
    hard = dp.build_pooling_model(prep, 0, mode="hard5")
    soft = dp.build_pooling_model(prep, 0, mode="soft5")
    hs, ss = hard.state_dict(), soft.state_dict()
    assert set(hs) == set(ss)
    assert all(torch.equal(hs[k], ss[k]) for k in hs), "arms must start from the identical init"
    assert sum(p.numel() for p in hard.parameters()) == 408_651
    assert sum(p.numel() for p in soft.parameters()) == 408_651
    assert isinstance(hard.distance_gate, torch.nn.Embedding)
    assert tuple(hard.distance_gate.weight.shape) == (5, 48)
    assert tuple(soft.distance_gate.weight.shape) == (5, 48)
    assert hard.pair_pooling == "hard5" and soft.pair_pooling == "soft5"
    # no trainable parameter / buffer encodes the pooling mode
    hard_keys = {name for name, _ in hard.named_parameters()}
    assert "pair_pooling" not in hard_keys
    assert not any("soft" in name.lower() for name in hard_keys)
    # building does not leak state dict keys with different shapes
    assert {k: tuple(v.shape) for k, v in hs.items()} == {k: tuple(v.shape) for k, v in ss.items()}


# ---------------------------------------------------------------------------
# statistics + decision mapping
# ---------------------------------------------------------------------------


def test_bootstrap_is_signed_centered_and_seed_deterministic():
    drv = _driver()
    rng = np.random.default_rng(0)
    d = rng.normal(loc=-0.01, scale=0.05, size=500)
    out = drv.bootstrap_paired(d, n=500, seed=0)
    again = drv.bootstrap_paired(d, n=500, seed=0)
    assert out == again
    assert out["ci95"][0] <= out["mean_diff"] <= out["ci95"][1]
    assert out["n_soft5_better"] + out["n_hard5_better"] + out["n_tied"] == len(d)
    positive = drv.bootstrap_paired(np.abs(d), n=200, seed=0)
    assert positive["mean_diff"] > 0


def test_decision_bands():
    drv = _driver()
    assert drv.decide_case(-0.02) == "A"
    assert drv.decide_case(-0.0100001) == "A"
    assert drv.decide_case(-0.010) == "A"
    assert drv.decide_case(-0.0099) == "B"
    assert drv.decide_case(-0.003) == "B"
    assert drv.decide_case(-0.0029) == "C"
    assert drv.decide_case(0.0) == "C"
    assert drv.decide_case(+0.01) == "C"


def test_resolve_decision_requires_all_predictive_conditions():
    drv = _driver()
    assert drv.resolve_decision("A", True, True, True, True) == "SOFT5_PREDICTIVE_SIGNAL"
    assert drv.resolve_decision("A", False, True, True, True) == "UNCERTAIN"
    for ordinary, not_extreme, engineering in ((False, True, True), (True, False, True), (True, True, False)):
        assert (
            drv.resolve_decision("A", True, ordinary, not_extreme, engineering)
            == "SOFT5_PREDICTIVE_SIGNAL_UNCERTAIN"
        )
    assert drv.resolve_decision("B", False, False, False, False) == "SOFT5_LIMITED_SIGNAL"
    assert drv.resolve_decision("C", True, True, True, True) == "SOFT5_NO_USEFUL_GAIN"


def test_subgroup_and_stratified_helpers():
    drv = _driver()
    n = 200
    y = np.linspace(-4.0, -1.0, n)
    errors = {
        "Hard5": np.full(n, 0.2),
        "Soft5": np.full(n, 0.1),
    }
    payload = drv._subgroup_report(
        y, errors, y <= -3.0, y > -3.0, "y<=-3", "y>-3"
    )
    assert payload["y<=-3"]["n"] + payload["y>-3"]["n"] == n
    for name in ("y<=-3", "y>-3"):
        assert abs(payload[name]["delta"] + 0.1) < 1e-12
        assert payload[name]["mae_hard5"] > payload[name]["mae_soft5"]
    strata = drv._stratified_report(
        np.arange(n, dtype=np.float64), errors, qs=(0.0, 0.5, 1.0), label="index"
    )
    assert len(strata["strata"]) >= 2
    assert strata["strata"][0]["range"][0] == 0.0
    assert strata["strata"][-1]["range"][1] == float(n - 1)
    for cell in strata["strata"]:
        assert cell["n"] > 0 and abs(cell["delta"] + 0.1) < 1e-12


def test_hashes_are_content_addressed():
    drv = _driver()
    a = {"x": torch.zeros(3), "y": torch.ones(2)}
    b = {"y": torch.ones(2), "x": torch.zeros(3)}
    assert drv.state_hash(a) == drv.state_hash(b)
    assert drv.state_hash(a) != drv.state_hash({"x": torch.zeros(3), "y": torch.ones(2) + 1})
    assert drv.sha256_rows(np.array([0, 1, 2])) == drv.sha256_rows(np.array([0, 1, 2]))
    assert drv.sha256_rows(np.array([0, 1, 2])) != drv.sha256_rows(np.array([0, 1, 3]))
