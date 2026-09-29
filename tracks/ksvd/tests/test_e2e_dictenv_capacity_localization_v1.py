"""Focused tests for ``e2e_dictenv_capacity_localization_v1`` (capacity round).

CPU only; the official ZINC test split is never touched.  The file covers the
frozen Stage-E list: CAP-BASE identity reproduction, CSSD-q1 / C6 invariance,
the three candidate architectures (F multi-rank fusion shapes and permutation
invariance, R single-pass pair composition without write-back, G gated
permutation-invariant summaries and empty-graph safety), the guards, the
parameter-budget rule, bit-identical warm loading of the CAP-BASE soup and
finite non-zero step-0 gradients of every new module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_capacity_localization_v1 as cl
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_capacity_localization_v1 as runner

CSSD_DIR = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1"
SOUP_PATH = CSSD_DIR / "training/checkpoints/CSSD-Q1-seed0_soup_state.pt"
SUBSPACE_PATH = CSSD_DIR / "common_subspace.json"


def _synthetic_phi(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.gamma(shape=2.0, scale=1.0, size=(n, cssd.PHI_DIM))
    base[:, 5] = 0.0
    base[:, 8] = 0.0
    return base


def _subspace(q: int = 1, seed: int = 0) -> cssd.CommonSubspace:
    return cssd.build_common_subspace(_synthetic_phi(seed=seed), q)


def _dictionary() -> np.ndarray:
    dictionary, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return dictionary


def _frozen_subspace() -> cssd.CommonSubspace:
    import json

    if not SUBSPACE_PATH.exists():  # pragma: no cover
        pytest.skip("CSSD subspace artifact not available")
    payload = json.loads(SUBSPACE_PATH.read_text())["q1"]
    return cssd.CommonSubspace(
        components=np.asarray(payload["components"], dtype=np.float64),
        rms=np.asarray(payload["rms"], dtype=np.float64),
        kind="q1",
    )


def _soup_state() -> dict[str, torch.Tensor]:
    if not SOUP_PATH.exists():  # pragma: no cover
        pytest.skip("CSSD seed-0 soup checkpoint not available")
    return torch.load(SOUP_PATH, map_location="cpu", weights_only=False)


def _mini_batches(count: int = 2, split: str = "train"):
    data = p1run.load_split(split, subset=count)
    if not data:  # pragma: no cover
        pytest.skip("ZINC encoded data not available")
    return data


# ---------------------------------------------------------------------------
# CAP-BASE identity
# ---------------------------------------------------------------------------


def test_cap_base_identity_reproduction() -> None:
    subspace = _subspace(seed=11)
    dictionary = _dictionary()
    torch.manual_seed(0)
    reference = cssd.build_cssd_model(dictionary, 0, subspace)
    torch.manual_seed(0)
    model = cl.build_capacity_model(dictionary, 0, subspace, "M0")
    reference_state = reference.state_dict()
    model_state = model.state_dict()
    assert set(reference_state) == set(model_state)
    for key in reference_state:
        assert torch.equal(reference_state[key], model_state[key]), key


def test_cap_base_forward_matches_cssd_model() -> None:
    subspace = _frozen_subspace()
    dictionary = _dictionary()
    data = _mini_batches(2)
    batch = p1.env_collate(data)
    torch.manual_seed(0)
    reference = cssd.build_cssd_model(dictionary, 0, subspace).eval()
    torch.manual_seed(0)
    model = cl.build_capacity_model(dictionary, 0, subspace, "M0").eval()
    with torch.no_grad():
        reference_prediction = reference(batch, mask=cssd.CSSD_MASK)
        candidate_prediction = model(batch, mask=cssd.CSSD_MASK)
    assert torch.equal(reference_prediction, candidate_prediction)


def test_m0_warm_soup_reproduces_frozen_cap_base_mae() -> None:
    subspace = _frozen_subspace()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "M0")
    model.load_state_dict(_soup_state())
    valid = p1run.load_split("valid")
    if not valid:  # pragma: no cover
        pytest.skip("valid data not available")
    loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    device = audit.attach_cpu(4)
    mae = cl.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
    assert abs(mae - cl.CAP_BASE_SOUP_MAE) < 5e-7


def test_cssd_q1_and_c6_unchanged() -> None:
    assert cm.c6_equivalence_check()
    assert cssd.CSSD_MASK is cm.C6_MASK
    assert cssd.CSSD_SPEC.mask_kind == "C6"
    assert cssd.CSSD_SPEC.node_binding == "paired"
    assert cssd.CSSD_SPEC.edge_binding == "paired"
    assert cssd.CSSD_SPEC.coding == "sparse"


def test_global_rng_stream_matches_cap_base() -> None:
    subspace = _subspace(seed=12)
    dictionary = _dictionary()
    torch.manual_seed(0)
    cssd.build_cssd_model(dictionary, 0, subspace)
    reference = torch.get_rng_state().clone()
    for kind in cl.KINDS:
        torch.manual_seed(0)
        cl.build_capacity_model(dictionary, 0, subspace, kind)
        assert torch.equal(torch.get_rng_state(), reference), kind


# ---------------------------------------------------------------------------
# warm start
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["M0", "F", "R", "G"])
def test_warm_state_loads_bit_identically(kind: str) -> None:
    subspace = _frozen_subspace()
    state = _soup_state()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, kind)
    report = cl.load_capacity_warm_state(model, state)
    assert report["bit_identical"]
    assert not report["unexpected_keys"]
    assert report["shared_keys"] == len(state)
    for key, value in state.items():
        target = model.state_dict()[key]
        if tuple(target.shape) == tuple(value.shape):
            assert torch.equal(target, value), key


@pytest.mark.parametrize("kind", ["F", "R", "G"])
def test_new_modules_receive_finite_nonzero_gradient(kind: str) -> None:
    subspace = _frozen_subspace()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, kind)
    cl.load_capacity_warm_state(model, _soup_state())
    batch = next(iter(p1.make_env_loader(_mini_batches(2), p2run.BATCH_SIZE, False, 0)))
    norms = cl.new_module_gradient_norms(model, batch, audit.attach_cpu(2))
    assert norms
    assert all(np.isfinite(value) for value in norms.values())
    assert all(value > 0.0 for value in norms.values()), norms


@pytest.mark.parametrize("kind", ["F", "R", "G"])
def test_capacity_off_reproduces_cap_base_predictions(kind: str) -> None:
    subspace = _frozen_subspace()
    state = _soup_state()
    base = cl.build_capacity_model(_dictionary(), 0, subspace, "M0")
    base.load_state_dict(state)
    model = cl.build_capacity_model(_dictionary(), 0, subspace, kind)
    cl.load_capacity_warm_state(model, state)
    model.capacity_off = True
    base.eval()
    model.eval()
    batch = p1.env_collate(_mini_batches(2))
    with torch.no_grad():
        base_prediction = base(batch, mask=cssd.CSSD_MASK)
        off_prediction = model(batch, mask=cssd.CSSD_MASK)
    assert torch.allclose(base_prediction, off_prediction, atol=1e-6, rtol=1e-5)

@pytest.mark.parametrize("kind", ["F", "R", "G"])
def test_step0_prediction_shift_is_not_catastrophic(kind: str) -> None:
    subspace = _frozen_subspace()
    state = _soup_state()
    valid = p1run.load_split("valid", subset=64)
    if not valid:  # pragma: no cover
        pytest.skip("valid data not available")
    loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    device = audit.attach_cpu(2)
    base = cl.build_capacity_model(_dictionary(), 0, subspace, "M0")
    base.load_state_dict(state)
    base_mae = cl.evaluate_mae(base, loader, device, cssd.CSSD_MASK)
    model = cl.build_capacity_model(_dictionary(), 0, subspace, kind)
    cl.load_capacity_warm_state(model, state)
    candidate_mae = cl.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
    assert abs(candidate_mae - base_mae) <= 0.03, (kind, candidate_mae, base_mae)


# ---------------------------------------------------------------------------
# candidate F — multi-rank structure-semantic fusion
# ---------------------------------------------------------------------------


def test_fusion_shapes_match_slot_widths() -> None:
    subspace = _frozen_subspace()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "F")
    batch = p1.env_collate(_mini_batches(2))
    q = torch.nn.functional.one_hot(batch.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).float()
    coord = model.code(batch.dict_phi)
    cu = coord[batch.env_bond_u]
    cv = coord[batch.env_bond_v]
    g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
    b = torch.nn.functional.one_hot(
        batch.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)
    ).float()
    node_occ = coord[batch.env_occ_node]
    qc = q[batch.env_occ_node]
    node = model._node_occurrence(node_occ, qc)
    edge = model._edge_occurrence(g, b)
    assert tuple(node.shape) == (int(batch.env_occ_node.shape[0]), int(p2.D_A))
    assert tuple(edge.shape) == (int(batch.env_bond_u.shape[0]), int(model.W_E_S.shape[1]))
    heads = model.head_products(coord, batch)
    assert tuple(heads["node"].shape) == (
        cl.FUSION_HEADS,
        int(batch.env_occ_node.shape[0]),
        cl.FUSION_HEAD_DIM,
    )
    assert tuple(heads["edge"].shape) == (
        cl.FUSION_HEADS,
        int(batch.env_bond_u.shape[0]),
        cl.FUSION_HEAD_DIM,
    )
    # structured, not a free MLP: both factors exist per head
    assert len(model.F_NS) == len(model.F_NC) == len(model.F_ES) == len(model.F_EC) == cl.FUSION_HEADS


def test_fusion_is_occurrence_permutation_invariant() -> None:
    subspace = _frozen_subspace()
    state = _soup_state()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "F")
    cl.load_capacity_warm_state(model, state)
    model.eval()
    data = _mini_batches(1)[0].clone()
    generator = torch.Generator().manual_seed(7)
    permutation = torch.randperm(int(data.env_occ_node.shape[0]), generator=generator)
    permuted = data.clone()
    permuted.env_occ_node = data.env_occ_node[permutation]
    permuted.env_occ_root = data.env_occ_root[permutation]
    permuted.env_occ_shell = data.env_occ_shell[permutation]
    batch_a = p1.env_collate([data])
    batch_b = p1.env_collate([permuted])
    with torch.no_grad():
        prediction_a = model(batch_a, mask=cssd.CSSD_MASK)
        prediction_b = model(batch_b, mask=cssd.CSSD_MASK)
    assert torch.allclose(prediction_a, prediction_b, atol=1e-5)


def test_fusion_structure_and_semantic_factors_both_receive_gradient() -> None:
    subspace = _frozen_subspace()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "F")
    batch = next(iter(p1.make_env_loader(_mini_batches(2), p2run.BATCH_SIZE, False, 0)))
    norms = cl.new_module_gradient_norms(model, batch, audit.attach_cpu(2))
    structure = [value for key, value in norms.items() if key.startswith(("F_NS", "F_ES"))]
    semantic = [value for key, value in norms.items() if key.startswith(("F_NC", "F_EC"))]
    assert structure and all(value > 0 for value in structure)
    assert semantic and all(value > 0 for value in semantic)


# ---------------------------------------------------------------------------
# candidate R — static relation-conditioned composition
# ---------------------------------------------------------------------------


def test_relation_pair_encoder_called_exactly_once() -> None:
    subspace = _frozen_subspace()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "R")
    batch = p1.env_collate(_mini_batches(2))
    counts = {"pair": 0, "relation": 0}

    def make_hook(name):
        def hook(module, inputs, output):  # noqa: ANN001
            counts[name] += 1

        return hook

    handles = [
        model.pair_encoder.register_forward_hook(make_hook("pair")),
        model.relation_encoder.register_forward_hook(make_hook("relation")),
    ]
    try:
        with torch.no_grad():
            model(batch, mask=cssd.CSSD_MASK)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"pair": 1, "relation": 1}


def test_relation_no_writeback_to_environments() -> None:
    subspace = _frozen_subspace()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "R")
    batch = p1.env_collate(_mini_batches(2))
    with torch.no_grad():
        _, aux_before = model(batch, mask=cssd.CSSD_MASK, return_aux=True)
    generator = torch.Generator().manual_seed(13)
    with torch.no_grad():
        for block in model.pair_blocks:
            for parameter in block.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.1
                )
        _, aux_after = model(batch, mask=cssd.CSSD_MASK, return_aux=True)
    assert torch.equal(aux_before["E"], aux_after["E"])
    assert torch.equal(aux_before["coord"], aux_after["coord"])


def test_relation_pair_order_invariance() -> None:
    subspace = _frozen_subspace()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "R")
    cl.load_capacity_warm_state(model, _soup_state())
    model.eval()
    data = _mini_batches(1)[0].clone()
    generator = torch.Generator().manual_seed(17)
    permutation = torch.randperm(int(data.pair_index.shape[1]), generator=generator)
    permuted = data.clone()
    permuted.pair_index = data.pair_index[:, permutation]
    permuted.pair_bucket = data.pair_bucket[permutation]
    permuted.pair_relation = data.pair_relation[permutation]
    batch_a = p1.env_collate([data])
    batch_b = p1.env_collate([permuted])
    with torch.no_grad():
        prediction_a = model(batch_a, mask=cssd.CSSD_MASK)
        prediction_b = model(batch_b, mask=cssd.CSSD_MASK)
    assert torch.allclose(prediction_a, prediction_b, atol=1e-5)


def test_relation_film_starts_as_identity_and_blocks_have_gradient() -> None:
    subspace = _frozen_subspace()
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "R")
    batch = next(iter(p1.make_env_loader(_mini_batches(2), p2run.BATCH_SIZE, False, 0)))
    norms = cl.new_module_gradient_norms(model, batch, audit.attach_cpu(2))
    assert len(model.pair_blocks) == cl.RELATION_BLOCKS
    assert all(value > 0 for value in norms.values())
    # FiLM at initialization: gamma/beta exactly zero -> identity modulation
    for block in model.pair_blocks:
        assert bool((block.gamma.weight == 0).all()) and bool((block.gamma.bias == 0).all())
        assert bool((block.beta.weight == 0).all()) and bool((block.beta.bias == 0).all())


# ---------------------------------------------------------------------------
# candidate G — gated invariant summaries
# ---------------------------------------------------------------------------


def test_gated_summary_node_and_pair_permutation_invariance() -> None:
    torch.manual_seed(0)
    gate = torch.nn.Linear(4, 6)
    value = torch.nn.Linear(4, 6)
    rows = 9
    values = torch.randn(rows, 4)
    batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
    summary = cl.ReadoutCapacityModel.gated_summary(values, batch, gate, value, 3)
    generator = torch.Generator().manual_seed(3)
    permutation = torch.randperm(rows, generator=generator)
    permuted = cl.ReadoutCapacityModel.gated_summary(values[permutation], batch[permutation], gate, value, 3)
    assert torch.allclose(summary, permuted, atol=1e-6)
    assert tuple(summary.shape) == (3, 6)


def test_gated_summary_empty_and_small_graph_safety() -> None:
    torch.manual_seed(1)
    gate = torch.nn.Linear(4, 5)
    value = torch.nn.Linear(4, 5)
    empty = cl.ReadoutCapacityModel.gated_summary(
        torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), gate, value, 4
    )
    assert tuple(empty.shape) == (4, 5)
    assert torch.isfinite(empty).all()
    assert bool((empty == 0).all())
    single = torch.randn(1, 4)
    one = cl.ReadoutCapacityModel.gated_summary(single, torch.zeros(1, dtype=torch.long), gate, value, 1)
    assert torch.isfinite(one).all()
    # a graph with zero contributing rows maps to an exact zero summary
    batch = torch.tensor([0, 1])
    rows = torch.randn(2, 4)
    pair_summary = cl.ReadoutCapacityModel.gated_summary(rows, batch, gate, value, 3)
    assert pair_summary.shape == (3, 5)
    assert bool((pair_summary[2] == 0).all())


def test_readout_reader_keeps_base_columns_bit_identical() -> None:
    subspace = _frozen_subspace()
    base = cl.build_capacity_model(_dictionary(), 0, subspace, "M0")
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "G")
    base_weight = base.reader.net[0].weight
    widened = model.reader.net[0].weight
    assert torch.equal(widened[:, : int(audit.READER_IN_DIM)], base_weight)
    assert torch.equal(model.reader.net[0].bias, base.reader.net[0].bias)
    assert torch.equal(model.reader.net[2].weight, base.reader.net[2].weight)
    assert torch.equal(model.reader.net[4].weight, base.reader.net[4].weight)
    assert tuple(widened.shape) == (int(base_weight.shape[0]), int(audit.READER_IN_DIM) + 2 * cl.READOUT_SUMMARY_DIM)


def test_readout_summary_disable_is_safe_and_individual() -> None:
    subspace = _frozen_subspace()
    state = _soup_state()
    base = cl.build_capacity_model(_dictionary(), 0, subspace, "M0")
    base.load_state_dict(state)
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "G")
    cl.load_capacity_warm_state(model, state)
    base.eval()
    model.eval()
    batch = p1.env_collate(_mini_batches(2))
    with torch.no_grad():
        base_prediction = base(batch, mask=cssd.CSSD_MASK)
        model.capacity_off = True
        off_prediction = model(batch, mask=cssd.CSSD_MASK)
        assert torch.allclose(off_prediction, base_prediction, atol=1e-6, rtol=1e-5)
        model.capacity_off = False
        deltas = {}
        for kind in ("env", "pair"):
            model.disabled_summaries = {kind}
            prediction = model(batch, mask=cssd.CSSD_MASK)
            assert torch.isfinite(prediction).all()
            deltas[kind] = float(torch.abs(prediction - base_prediction).max())
        model.disabled_summaries = set()
    assert deltas["env"] > 0.0 and deltas["pair"] > 0.0


# ---------------------------------------------------------------------------
# guards and budget
# ---------------------------------------------------------------------------


def test_cpu_only_guard() -> None:
    cl.cpu_only_guard(torch.device("cpu"))
    with pytest.raises(RuntimeError):
        cl.cpu_only_guard(torch.device("cuda"))


def test_official_test_blocker() -> None:
    cl.official_test_blocker({"official_test_loaded": False})
    with pytest.raises(RuntimeError):
        cl.official_test_blocker({"official_test_loaded": True})


def test_parameter_budget_rule() -> None:
    payload = cl.parameter_budget(_subspace(seed=21))
    assert payload["cap_base_params"] == 97727
    assert payload["added_params_ratio_ok"]
    assert payload["added_params_ratio"] <= cl.BUDGET_RATIO_MAX
    for kind in ("F", "R", "G"):
        row = payload["rows"][kind]
        assert row["added_params_vs_cap_base"] > 0
        assert row["within_hard_ceiling"]
        assert cl.BUDGET_PREFERRED[0] <= row["total_params"] <= cl.BUDGET_PREFERRED[1]
    assert payload["candidate_added_params"]["F"] == 29568
    assert payload["candidate_added_params"]["R"] == 34400
    assert payload["candidate_added_params"]["G"] == 30336
    assert payload["official_test_loaded"] is False


def test_gate_and_winner_rules() -> None:
    control = {"soup_valid_mae": 0.130, "last10_mean_valid_mae": 0.131}
    rows = {
        "M0": control,
        "F": {"soup_valid_mae": 0.1235, "last10_mean_valid_mae": 0.127},
        "R": {"soup_valid_mae": 0.1260, "last10_mean_valid_mae": 0.127},
        "G": {"soup_valid_mae": 0.140, "last10_mean_valid_mae": 0.141},
    }
    deltas = cl.screening_deltas(rows)
    gates = {kind: cl.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    assert gates["F"]["passed"] and gates["F"]["verdict"] == "CAPACITY_SIGNAL"
    assert gates["R"]["passed"]
    assert not gates["G"]["passed"]
    selection = cl.select_winner(rows, gates)
    assert selection["winner"] == "F"
    assert selection["reason"] == "SCREENING_SOUP_MAE"
    # within 0.002 the frozen F > R > G tie-break decides
    rows["R"]["soup_valid_mae"] = 0.1248
    rows["R"]["last10_mean_valid_mae"] = 0.1274
    deltas = cl.screening_deltas(rows)
    gates = {kind: cl.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    selection = cl.select_winner(rows, gates)
    assert selection["winner"] == "F"
    assert selection["reason"] == "TIE_BREAK_F_R_G"
    # nobody passes the frozen gate -> no purchase
    rows["F"]["soup_valid_mae"] = 0.130
    rows["F"]["last10_mean_valid_mae"] = 0.131
    rows["R"]["soup_valid_mae"] = 0.131
    rows["R"]["last10_mean_valid_mae"] = 0.132
    rows["G"]["soup_valid_mae"] = 0.129
    rows["G"]["last10_mean_valid_mae"] = 0.130
    deltas = cl.screening_deltas(rows)
    gates = {kind: cl.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    selection = cl.select_winner(rows, gates)
    assert selection["winner"] is None
    assert selection["reason"] == "NO_CLEAR_CAPACITY_LOCALIZATION"


def test_full_interpretation_bands() -> None:
    assert cl.full_interpretation(0.130)["band"] == "FULL_CAPACITY_GAIN_NOT_ESTABLISHED"
    assert cl.full_interpretation(0.124)["band"] == "CAPACITY_DIRECTION_SUPPORTED_SINGLE_SEED"
    assert cl.full_interpretation(0.119)["band"] == "NEW_PERFORMANCE_BAND_SINGLE_SEED"
    assert cl.full_interpretation(0.109)["band"] == "MAJOR_CAPACITY_BOTTLENECK_IDENTIFIED"


# ---------------------------------------------------------------------------
# screening loop integration (tiny)
# ---------------------------------------------------------------------------


def test_train_screen_smoke_on_tiny_subset() -> None:
    subspace = _frozen_subspace()
    state = _soup_state()
    train = p1run.load_split("train", subset=8)
    valid = p1run.load_split("valid", subset=8)
    if not train or not valid:  # pragma: no cover
        pytest.skip("ZINC encoded data not available")
    model = cl.build_capacity_model(_dictionary(), 0, subspace, "G")
    cl.load_capacity_warm_state(model, state)
    payload = cl.train_screen(
        tag="SMOKE",
        model=model,
        dictionary=_dictionary(),
        subspace=subspace,
        epochs=2,
        threads=2,
        train_data=train,
        valid_data=valid,
        seed=0,
        soup_window=(1, 2),
        soup_k=2,
        log=False,
    )
    assert payload["epochs_run"] == 2
    assert len(payload["soup_members"]) == 2
    assert np.isfinite(payload["soup_valid_mae"])
    assert np.isfinite(payload["last10_mean_valid_mae"])
    assert payload["official_test_loaded"] is False


# ---------------------------------------------------------------------------
# runner plumbing (imports, io helpers, frozen early-stop callback)
# ---------------------------------------------------------------------------


def test_runner_import_and_arm_layout() -> None:
    assert set(runner.ARM_DIRS) == set(cl.KINDS)
    assert tuple(runner.PROBES) == (
        "P1_phi_row_shuffle",
        "P2_node_correspondence",
        "P3_edge_correspondence",
        "P4_relation",
    )
    assert runner.PROTOCOL_VERSION == cl.PROTOCOL_VERSION
    assert runner.ARM_DIRS["M0"].name == "m0"
    assert runner.ARM_DIRS["G"].name == "readout"


def test_runner_csv_round_trip(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "table.csv"
    runner._write_csv(path, rows, ["a", "b", "missing"])
    read = runner._read_csv_rows(path)
    assert [row["a"] for row in read] == ["1", "2"]
    assert [row["b"] for row in read] == ["x", "y"]
    assert read[0]["missing"] == ""


def test_runner_fmt() -> None:
    assert runner._fmt(0.123456789) == "0.123457"
    assert runner._fmt(1) == "1.000000"
    assert runner._fmt(None) == "None"


class _FakeModel:
    def __init__(self) -> None:
        self.D = torch.zeros(1)

    def named_parameters(self):
        return iter(())


def test_runner_early_stop_callback_guards() -> None:
    model = _FakeModel()
    reference = [0.10] * 320
    callback = runner._full_early_stop_callback(reference, [])
    keep, payload = callback(1, model, None, [{"epoch": 1, "train_mae": float("nan"), "valid_mae": 0.1}])
    assert keep is False and payload["reason"] == "non_finite"
    callback = runner._full_early_stop_callback(reference, [])
    keep, payload = callback(
        1, model, None, [{"epoch": 1, "train_mae": 0.1, "valid_mae": float(cssd.CATASTROPHIC_MAE) + 1.0}]
    )
    assert keep is False and payload["reason"] == "divergence"


def test_runner_early_stop_callback_streak_and_trend() -> None:
    model = _FakeModel()
    reference = [0.10] * 320
    # a constant +0.40 gap with no improving trend stops exactly on epoch 40
    callback = runner._full_early_stop_callback(reference, [])
    stopped: int | None = None
    for epoch in range(1, 61):
        keep, payload = callback(epoch, model, None, [{"epoch": epoch, "train_mae": 0.1, "valid_mae": 0.5}])
        if not keep:
            stopped = epoch
            assert payload["reason"] == "matched_cssd_curve_hopeless"
            assert payload["worse_streak"] >= 40
            break
    assert stopped == 40
    # an arm matching the reference curve is never stopped
    callback = runner._full_early_stop_callback(reference, [])
    for epoch in range(1, 121):
        keep, _payload = callback(epoch, model, None, [{"epoch": epoch, "train_mae": 0.1, "valid_mae": 0.1}])
        assert keep
