"""Focused tests for ``e2e_dictenv_capacity_localization_v2``.

The v2 round repairs the warm-adaptation screen of v1 (calibrated M0 at
``Adam(lr = 1e-4)``, differential LR ``base 1e-4 / new 1e-3``, near-zero but
live residual initialisation) while keeping the v1 F/R/G architectures frozen.

CPU only; the official ZINC test split is never touched.  The file covers the
frozen list: CAP-BASE identity reproduction, exact trainable-parameter
partition, differential-LR optimizer groups, v1 architecture equality, the
residual-zero == CAP-BASE identity for F/R/G, step-0 shift accounting, finite
non-zero new-branch gradients, the calibration gate, the S1-S4 screen gate with
the absolute ``0.127`` anchor, winner selection / tie-break, the full-run
bands, the guards and the tiny warm-loop smoke test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_capacity_localization_v1 as cl
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_capacity_localization_v2 as cv2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_capacity_localization_v2 as runner
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

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


def _base_prediction_and_loader(valid_size: int = 64):
    subspace = _frozen_subspace()
    state = _soup_state()
    valid = p1run.load_split("valid", subset=valid_size)
    if not valid:  # pragma: no cover
        pytest.skip("valid data not available")
    loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    device = audit.attach_cpu(2)
    base = cv2.build_v2_model(_dictionary(), 0, subspace, "M0")
    base.load_state_dict(state)
    base.eval()
    prediction = cv2.predictions_for(base, loader, device, cssd.CSSD_MASK)
    return subspace, state, loader, device, base, prediction


# ---------------------------------------------------------------------------
# CAP-BASE identity and configuration invariants
# ---------------------------------------------------------------------------


def test_cap_base_identity_reproduction() -> None:
    subspace = _subspace(seed=11)
    dictionary = _dictionary()
    torch.manual_seed(0)
    reference = cssd.build_cssd_model(dictionary, 0, subspace)
    torch.manual_seed(0)
    model = cv2.build_v2_model(dictionary, 0, subspace, "M0")
    reference_state = reference.state_dict()
    model_state = model.state_dict()
    assert set(reference_state) == set(model_state)
    for key in reference_state:
        assert torch.equal(reference_state[key], model_state[key]), key


def test_m0_warm_soup_reproduces_frozen_cap_base_mae() -> None:
    subspace = _frozen_subspace()
    model = cv2.build_v2_model(_dictionary(), 0, subspace, "M0")
    model.load_state_dict(_soup_state())
    valid = p1run.load_split("valid")
    if not valid:  # pragma: no cover
        pytest.skip("valid data not available")
    loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    device = audit.attach_cpu(4)
    mae = cv2.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
    assert abs(mae - cv2.CAP_BASE_SOUP_MAE) < 5e-7


def test_frozen_protocol_constants() -> None:
    assert cv2.CALIBRATION_LR == 1.0e-4
    assert cv2.CALIBRATION_EPOCHS == 20
    assert cv2.BASE_LR == 1.0e-4
    assert cv2.NEW_LR == 1.0e-3
    assert cv2.WEIGHT_DECAY == 1.0e-5
    assert cv2.SCREEN_EPOCHS == 40
    assert cv2.SCREEN_SOUP_WINDOW == (21, 40)
    assert cv2.CALIB_SOUP_WINDOW == (1, 20)
    assert cv2.CALIB_LATE_WINDOW == (16, 20)
    assert cv2.S2_ABS_MAX == 0.1270
    assert cv2.S1_DELTA_VS_M0 == -0.003
    assert cv2.S3_DELTA_LAST10_VS_M0 == -0.003
    assert cv2.STRONG_SCREEN_MAX == 0.123
    assert cv2.FULL_EPOCHS == 320
    assert cv2.FULL_LR == 1.0e-3
    assert cv2.TIE_ORDER == ("F", "R", "G")
    assert cv2.INIT_SHIFT_HARD_MAX == 0.002
    assert cv2.INIT_SHIFT_PREFERRED_MAX == 0.001
    assert cv2.CAP_BASE_SOUP_MAE == 0.13002798487985273


def test_cssd_q1_and_c6_unchanged() -> None:
    assert cm.c6_equivalence_check()
    assert cssd.CSSD_MASK is cm.C6_MASK
    assert cssd.CSSD_SPEC.mask_kind == "C6"
    assert cssd.CSSD_SPEC.node_binding == "paired"
    assert cssd.CSSD_SPEC.edge_binding == "paired"


def test_global_rng_stream_matches_cap_base() -> None:
    subspace = _subspace(seed=12)
    dictionary = _dictionary()
    torch.manual_seed(0)
    cssd.build_cssd_model(dictionary, 0, subspace)
    reference = torch.get_rng_state().clone()
    for kind in cv2.KINDS:
        torch.manual_seed(0)
        cv2.build_v2_model(dictionary, 0, subspace, kind)
        assert torch.equal(torch.get_rng_state(), reference), kind


# ---------------------------------------------------------------------------
# parameter partition and differential-LR optimizer groups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["M0", "F", "R", "G"])
def test_parameter_partition_is_exact(kind: str) -> None:
    model = cv2.build_v2_model(_dictionary(), 0, _subspace(seed=13), kind)
    partition = cv2.parameter_partition(model)
    known = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert sorted(partition["all"]) == sorted(known)
    assert not set(partition["base"]) & set(partition["new"])
    assert len(partition["all"]) == len(set(partition["all"]))


@pytest.mark.parametrize("kind", ["M0", "F", "R", "G"])
def test_optimizer_groups_partition_and_rates(kind: str) -> None:
    model = cv2.build_v2_model(_dictionary(), 0, _subspace(seed=14), kind)
    optimizer, report = cv2.build_warm_optimizer(model)
    parameters = dict(model.named_parameters())
    seen: list[int] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            seen.append(id(parameter))
    assert len(seen) == len(set(seen)) == len(parameters)
    assert report["base_lr"] == cv2.BASE_LR
    assert report["new_lr"] == cv2.NEW_LR
    assert report["base_parameter_count"] + report["new_parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    by_lr = {float(group["lr"]): group for group in optimizer.param_groups}
    if kind == "M0":
        assert report["new_parameters"] == []
        assert report["new_parameter_count"] == 0
        assert report["group_count"] == 1
        assert list(by_lr) == [cv2.BASE_LR]
        assert len(by_lr[cv2.BASE_LR]["params"]) == len(parameters)
    else:
        assert report["new_parameter_count"] > 0
        assert report["base_parameter_count"] == 97727
        assert report["group_count"] == 2
        assert set(by_lr) == {cv2.BASE_LR, cv2.NEW_LR}
        base_names = {
            name for name, parameter in model.named_parameters() if id(parameter) in {id(p) for p in by_lr[cv2.BASE_LR]["params"]}
        }
        assert base_names == set(report["base_parameters"])
        assert all(
            not any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in model.CAPACITY_NEW_PREFIXES
            )
            for name in base_names
        )


def test_m0_optimizer_rate_is_calibration_rate() -> None:
    model = cv2.build_v2_model(_dictionary(), 0, _subspace(seed=15), "M0")
    optimizer, report = cv2.build_warm_optimizer(
        model, base_lr=cv2.CALIBRATION_LR, new_lr=cv2.CALIBRATION_LR
    )
    assert [float(group["lr"]) for group in optimizer.param_groups] == [cv2.CALIBRATION_LR]
    assert report["base_lr"] == report["new_lr"] == 1.0e-4


def test_added_parameter_counts_equal_v1() -> None:
    subspace = _subspace(seed=16)
    dictionary = _dictionary()
    for kind in ("F", "R", "G"):
        v1_model = cl.build_capacity_model(dictionary, 0, subspace, kind)
        v2_model = cv2.build_v2_model(dictionary, 0, subspace, kind)
        v1_base = sum(p.numel() for p in cl.build_capacity_model(dictionary, 0, subspace, "M0").parameters())
        v1_added = sum(p.numel() for p in v1_model.parameters()) - v1_base
        # v1's ``capacity_parameter_count`` omits G's appended reader columns;
        # v2 counts them (they are new parameters), so compare against the
        # total-minus-base figure the v1 budget also used.
        assert v2_model.capacity_parameter_count() == v1_added
        assert v2_model.capacity_parameter_count() == runner.V1_ADDED_PARAMS[kind]
        assert sum(p.numel() for p in v2_model.parameters()) == sum(
            p.numel() for p in v1_model.parameters()
        )
        assert sum(p.numel() for p in v2_model.parameters()) == runner.V1_TOTAL_PARAMS[kind]
    m0 = cv2.build_v2_model(dictionary, 0, subspace, "M0")
    assert sum(p.numel() for p in m0.parameters()) == runner.V1_TOTAL_PARAMS["M0"]


# ---------------------------------------------------------------------------
# v1 architecture equality (only the residual projection scaling changed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["M0", "F", "R", "G"])
def test_v2_architecture_matches_v1(kind: str) -> None:
    subspace = _subspace(seed=17)
    dictionary = _dictionary()
    torch.manual_seed(0)
    v1_model = cl.build_capacity_model(dictionary, 0, subspace, kind)
    torch.manual_seed(0)
    v2_model = cv2.build_v2_model(dictionary, 0, subspace, kind)
    v1_shapes = {name: tuple(tensor.shape) for name, tensor in v1_model.state_dict().items()}
    v2_shapes = {name: tuple(tensor.shape) for name, tensor in v2_model.state_dict().items()}
    if kind == "G":
        # the only structural difference: v1 stores the appended reader columns
        # inside the widened first layer, v2 keeps them as an explicit
        # bias-free projection (same number of parameters).
        assert set(v1_shapes) == set(v2_shapes) - {"reader.summary_proj.weight"}
        appended = v2_shapes["reader.summary_proj.weight"]
        assert appended == (
            v1_shapes["reader.net.0.weight"][0],
            2 * cl.READOUT_SUMMARY_DIM,
        )
    else:
        assert set(v1_shapes) == set(v2_shapes)
    for name, shape in v1_shapes.items():
        if kind == "G" and name == "reader.net.0.weight":
            assert shape[0] == v2_shapes[name][0]
            assert v2_shapes[name][1] == shape[1] - 2 * cl.READOUT_SUMMARY_DIM
            continue
        assert shape == v2_shapes[name], name
    assert sum(p.numel() for p in v2_model.parameters()) == sum(
        p.numel() for p in v1_model.parameters()
    )


def test_f_architecture_scaling_only() -> None:
    subspace = _subspace(seed=18)
    dictionary = _dictionary()
    torch.manual_seed(0)
    v1_model = cl.build_capacity_model(dictionary, 0, subspace, "F")
    torch.manual_seed(0)
    v2_model = cv2.build_v2_model(dictionary, 0, subspace, "F")
    factor = cv2.FUSION_RESIDUAL_SCALE / cl.FUSION_PROJ_INIT
    for name, tensor in v1_model.state_dict().items():
        target = v2_model.state_dict()[name]
        if name in ("F_NP.weight", "F_EP.weight"):
            assert torch.equal(target, tensor * factor), name
            continue
        assert torch.equal(target, tensor), name
    assert len(v2_model.F_NS) == len(v2_model.F_NC) == cl.FUSION_HEADS
    assert all(parameter.out_features == cl.FUSION_HEAD_DIM for parameter in v2_model.F_NS)


def test_r_architecture_scaling_only() -> None:
    subspace = _subspace(seed=19)
    dictionary = _dictionary()
    torch.manual_seed(0)
    v1_model = cl.build_capacity_model(dictionary, 0, subspace, "R")
    torch.manual_seed(0)
    v2_model = cv2.build_v2_model(dictionary, 0, subspace, "R")
    factor = cv2.RELATION_RESIDUAL_SCALE / cl.RELATION_RESIDUAL_INIT
    for name, tensor in v1_model.state_dict().items():
        target = v2_model.state_dict()[name]
        scaled = name.startswith("pair_blocks") and name.endswith(("fc2.weight", "fc2.bias"))
        if scaled:
            assert torch.equal(target, tensor * factor), name
            continue
        assert torch.equal(target, tensor), name
    assert len(v2_model.pair_blocks) == cl.RELATION_BLOCKS
    assert all(block.fc2.out_features == int(p2.PAIR_HIDDEN) for block in v2_model.pair_blocks)
    assert all(
        block.fc1.out_features == cl.RELATION_HIDDEN for block in v2_model.pair_blocks
    )
    for block in v2_model.pair_blocks:
        assert bool((block.gamma.weight == 0).all()) and bool((block.beta.weight == 0).all())


def test_g_architecture_is_functionally_equal_to_v1_g() -> None:
    """Transplant v1's summary parameters into v2 and compare predictions.

    The two implementations differ only in how the appended reader columns are
    held (one widened first layer vs a separate bias-free projection), so with
    identical parameters they must compute the same function up to float
    summation order.
    """
    subspace = _frozen_subspace()
    dictionary = _dictionary()
    torch.manual_seed(0)
    v1_model = cl.build_capacity_model(dictionary, 0, subspace, "G")
    torch.manual_seed(0)
    v2_model = cv2.build_v2_model(dictionary, 0, subspace, "G")
    base_in = int(audit.READER_IN_DIM)
    assert torch.equal(v2_model.reader.net[0].weight, v1_model.reader.net[0].weight[:, :base_in])
    assert torch.equal(v2_model.reader.net[0].bias, v1_model.reader.net[0].bias)
    assert torch.equal(v2_model.reader.net[2].weight, v1_model.reader.net[2].weight)
    assert torch.equal(v2_model.reader.net[4].weight, v1_model.reader.net[4].weight)
    with torch.no_grad():
        for name in (
            "summary_gate_env",
            "summary_value_env",
            "summary_gate_pair",
            "summary_value_pair",
        ):
            getattr(v2_model, name).weight.copy_(getattr(v1_model, name).weight)
            getattr(v2_model, name).bias.copy_(getattr(v1_model, name).bias)
        v2_model.reader.summary_proj.weight.copy_(v1_model.reader.net[0].weight[:, base_in:])
    batch = p1.env_collate(_mini_batches(2))
    v1_model.eval()
    v2_model.eval()
    with torch.no_grad():
        v1_prediction = v1_model(batch, mask=cssd.CSSD_MASK)
        v2_prediction = v2_model(batch, mask=cssd.CSSD_MASK)
    assert torch.allclose(v1_prediction, v2_prediction, atol=1e-5, rtol=1e-6)
    # the raw initialisations draw the same multiset of appended-column weights
    assert torch.equal(
        torch.sort(v2_model.reader.summary_proj.weight.flatten()).values,
        torch.sort(v1_model.reader.net[0].weight[:, base_in:].flatten()).values,
    )


# ---------------------------------------------------------------------------
# residual-zero identity and step-0 accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["M0", "F", "R", "G"])
def test_warm_state_loads_bit_identically(kind: str) -> None:
    subspace = _frozen_subspace()
    state = _soup_state()
    model = cv2.build_v2_model(_dictionary(), 0, subspace, kind)
    report = cv2.load_capacity_warm_state(model, state)
    assert report["bit_identical"]
    assert not report["unexpected_keys"]
    assert report["shared_keys"] == len(state)
    for key, value in state.items():
        target = model.state_dict()[key]
        if tuple(target.shape) == tuple(value.shape):
            assert torch.equal(target, value), key


@pytest.mark.parametrize("kind", ["F", "R", "G"])
def test_zero_residual_reproduces_cap_base_predictions_exactly(kind: str) -> None:
    subspace, state, loader, device, base, base_prediction = _base_prediction_and_loader()
    model = cv2.build_v2_model(_dictionary(), 0, subspace, kind)
    cv2.load_capacity_warm_state(model, state)
    model.eval()
    model.zero_residual()
    prediction = cv2.predictions_for(model, loader, device, cssd.CSSD_MASK)
    assert torch.equal(torch.from_numpy(prediction), torch.from_numpy(base_prediction))


@pytest.mark.parametrize("kind", ["F", "R", "G"])
def test_capacity_off_reproduces_cap_base_predictions(kind: str) -> None:
    subspace, state, loader, device, base, base_prediction = _base_prediction_and_loader()
    model = cv2.build_v2_model(_dictionary(), 0, subspace, kind)
    cv2.load_capacity_warm_state(model, state)
    model.eval()
    model.capacity_off = True
    prediction = cv2.predictions_for(model, loader, device, cssd.CSSD_MASK)
    assert np.allclose(prediction, base_prediction, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kind", ["M0", "F", "R", "G"])
def test_step0_prediction_shift_bounds_and_accounting(kind: str) -> None:
    subspace, state, loader, device, base, base_prediction = _base_prediction_and_loader()
    model = cv2.build_v2_model(_dictionary(), 0, subspace, kind)
    cv2.load_capacity_warm_state(model, state)
    model.eval()
    prediction = cv2.predictions_for(model, loader, device, cssd.CSSD_MASK)
    delta = np.abs(prediction - base_prediction)
    # the helper used by the runner must report exactly these statistics
    reported = cv2.prediction_shift(model, loader, device, base_prediction)
    assert reported["mean_abs_prediction_delta"] == pytest.approx(float(delta.mean()), abs=0.0)
    assert reported["max_abs_prediction_delta"] == pytest.approx(float(delta.max()), abs=0.0)
    if kind == "M0":
        assert float(delta.max()) == 0.0
    else:
        assert float(delta.mean()) <= cv2.INIT_SHIFT_HARD_MAX
    assert np.isfinite(delta).all()


@pytest.mark.parametrize("kind", ["F", "R", "G"])
def test_new_modules_receive_finite_nonzero_gradient(kind: str) -> None:
    subspace = _frozen_subspace()
    model = cv2.build_v2_model(_dictionary(), 0, subspace, kind)
    cv2.load_capacity_warm_state(model, _soup_state())
    batch = next(iter(p1.make_env_loader(_mini_batches(2), p2run.BATCH_SIZE, False, 0)))
    norms = cv2.new_module_gradient_norms(model, batch, audit.attach_cpu(2))
    assert norms
    assert all(np.isfinite(value) for value in norms.values())
    assert all(value > 0.0 for value in norms.values()), norms


def test_step0_shift_is_smaller_than_v1_for_f_and_r() -> None:
    subspace, state, loader, device, base, base_prediction = _base_prediction_and_loader()
    for kind in ("F", "R"):
        model = cv2.build_v2_model(_dictionary(), 0, subspace, kind)
        cv2.load_capacity_warm_state(model, state)
        model.eval()
        shift = cv2.prediction_shift(model, loader, device, base_prediction)
        assert shift["mean_abs_prediction_delta"] < runner.V1_STEP0_MEAN_SHIFT[kind]


# ---------------------------------------------------------------------------
# calibration gate
# ---------------------------------------------------------------------------


def _calibration_result(
    *,
    soup: float,
    epoch_valid: list[float],
    best: float | None = None,
) -> dict:
    curve = [
        {"epoch": index + 1, "valid_mae": float(value)}
        for index, value in enumerate(epoch_valid)
    ]
    return {
        "soup_valid_mae": float(soup),
        "best_valid_mae": float(min(row["valid_mae"] for row in curve)) if best is None else float(best),
        "soup_members": [1, 2, 3, 4, 5],
        "curve": curve,
    }


def test_calibration_gate_passes_on_a_stable_curve() -> None:
    curve = [0.1300 + 0.0002 * index for index in range(20)]
    result = _calibration_result(soup=0.1297, epoch_valid=curve)
    summary = cv2.calibration_summary(result)
    assert summary["C1_pass"] and summary["C2_pass"] and summary["C3_pass"]
    assert summary["overall_pass"]
    assert summary["verdict"] == "WARM_ADAPTATION_PROTOCOL_VALIDATED"
    assert summary["calibration_last5"] == pytest.approx(float(np.mean(curve[15:])), abs=1e-12)
    assert summary["epoch20"] == pytest.approx(curve[19], abs=1e-12)
    assert summary["delta_soup"] == pytest.approx(0.1297 - cv2.CAP_BASE_SOUP_MAE, abs=1e-12)
    assert summary["late_slope"] == pytest.approx(2.0e-4, rel=1e-6)


def test_calibration_gate_fails_on_degradation() -> None:
    # C1 failure: soup above 0.1320
    curve = [0.1300 + 0.0001 * index for index in range(20)]
    summary = cv2.calibration_summary(_calibration_result(soup=0.1325, epoch_valid=curve))
    assert not summary["C1_pass"] and not summary["overall_pass"]
    assert summary["verdict"] == "WARM_ADAPTATION_PROTOCOL_UNSTABLE"
    # C2 failure: late window mean above 0.1350
    curve = [0.1300] * 15 + [0.1360] * 5
    summary = cv2.calibration_summary(_calibration_result(soup=0.1305, epoch_valid=curve))
    assert summary["C1_pass"] and not summary["C2_pass"] and not summary["overall_pass"]
    # C3 failure: epoch 20 above 0.137
    curve = [0.1300] * 19 + [0.1375]
    summary = cv2.calibration_summary(_calibration_result(soup=0.1305, epoch_valid=curve))
    assert summary["C2_pass"] and not summary["C3_pass"] and not summary["overall_pass"]
    # C3 failure: late divergence slope
    curve = [0.1300] * 15 + [0.1300, 0.1310, 0.1320, 0.1330, 0.1340]
    summary = cv2.calibration_summary(_calibration_result(soup=0.1305, epoch_valid=curve))
    assert summary["late_slope"] == pytest.approx(1.0e-3, rel=1e-6)
    assert not summary["C3_pass"]


# ---------------------------------------------------------------------------
# screen gate, absolute anchor and winner selection
# ---------------------------------------------------------------------------


def _screen_row(
    *,
    soup: float,
    last10: float,
    branch_grad: float = 1.0e-3,
    branch_update: float = 1.0e-4,
) -> dict:
    return {
        "soup_valid_mae": float(soup),
        "last_mean_valid_mae": float(last10),
        "branch_grad_max": float(branch_grad),
        "branch_update_max": float(branch_update),
        "branch_grad_finite": True,
        "branch_update_finite": True,
    }


def test_screen_gate_requires_all_four_conditions() -> None:
    control = _screen_row(soup=0.1310, last10=0.1330)
    rows = {
        "M0": control,
        # passes S1 (-0.005), S2 (0.1260 <= 0.1270), S3 (-0.004), S4
        "F": _screen_row(soup=0.1260, last10=0.1290),
        # passes S1 (-0.0035) but fails S2 only (0.1275 > 0.1270)
        "R": _screen_row(soup=0.1275, last10=0.1290),
        # passes S1/S2/S3 but fails S4 only (zero branch update)
        "G": _screen_row(soup=0.1240, last10=0.1260, branch_update=0.0),
    }
    deltas = cv2.screening_deltas(rows)
    gates = {kind: cv2.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    assert gates["F"]["passed"] and gates["F"]["verdict"] == "CAPACITY_SIGNAL"
    assert gates["R"]["S1_matched_control"] is True
    assert gates["R"]["S2_absolute"] is False and not gates["R"]["passed"]
    assert gates["G"]["S1_matched_control"]
    assert gates["G"]["S2_absolute"] and gates["G"]["S3_late_window"]
    assert gates["G"]["S4_branch_usage"] is False and not gates["G"]["passed"]
    selection = cv2.select_winner(rows, gates)
    assert selection["winner"] == "F"
    assert selection["reason"] == "SCREENING_SOUP_MAE"
    assert deltas["F"]["delta_soup_vs_M0"] == pytest.approx(-0.005, abs=1e-12)
    assert deltas["F"]["delta_soup_vs_start"] == pytest.approx(0.1260 - cv2.CAP_BASE_SOUP_MAE, abs=1e-12)
    assert deltas["F"]["delta_last10_vs_start"] == pytest.approx(
        0.1290 - cv2.CAP_BASE_SOUP_MAE, abs=1e-12
    )


def test_screen_gate_absolute_anchor_and_strong_band() -> None:
    control = _screen_row(soup=0.1310, last10=0.1330)
    rows = {
        "M0": control,
        # -0.009 matched delta and 0.1220 <= 0.123 -> STRONG
        "F": _screen_row(soup=0.1220, last10=0.1260),
        "R": _screen_row(soup=0.1400, last10=0.1400),
        "G": _screen_row(soup=0.1400, last10=0.1400),
    }
    deltas = cv2.screening_deltas(rows)
    gates = {kind: cv2.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    assert gates["F"]["S1_matched_control"] and gates["F"]["S2_absolute"] and gates["F"]["S3_late_window"]
    assert gates["F"]["verdict"] == "STRONG_CAPACITY_SIGNAL"
    assert gates["F"]["passed"]
    # a candidate that improves on M0 but not on the absolute anchor must fail
    rows["R"] = _screen_row(soup=0.1278, last10=0.1300)
    deltas = cv2.screening_deltas(rows)
    gate = cv2.capacity_gate(rows["R"], deltas["R"])
    assert gate["S1_matched_control"] and not gate["S2_absolute"] and not gate["passed"]
    # a candidate that is absolutely good but not better than its matched M0
    # must fail (S2 alone is not sufficient)
    rows["M0"] = _screen_row(soup=0.1265, last10=0.1285)
    rows["G"] = _screen_row(soup=0.1268, last10=0.1290)
    deltas = cv2.screening_deltas(rows)
    gate = cv2.capacity_gate(rows["G"], deltas["G"])
    assert gate["S2_absolute"] and not gate["S1_matched_control"] and not gate["passed"]


def test_screen_gate_last10_confirmation() -> None:
    control = _screen_row(soup=0.1295, last10=0.1310)
    rows = {
        "M0": control,
        "F": _screen_row(soup=0.1260, last10=0.1305),  # S1/S2 pass, S3 fails
        "R": _screen_row(soup=0.1400, last10=0.1400),
        "G": _screen_row(soup=0.1400, last10=0.1400),
    }
    deltas = cv2.screening_deltas(rows)
    gate = cv2.capacity_gate(rows["F"], deltas["F"])
    assert gate["S1_matched_control"] and gate["S2_absolute"] and not gate["S3_late_window"] and not gate["passed"]


def test_no_winner_reason_and_tie_break() -> None:
    control = _screen_row(soup=0.1295, last10=0.1310)
    # outside the 0.002 tolerance the lowest screening soup wins, even when it
    # is the last-priority candidate (G)
    rows = {
        "M0": control,
        "F": _screen_row(soup=0.1270, last10=0.1280),  # -0.0025 -> S1 fails
        "R": _screen_row(soup=0.1260, last10=0.1275),
        "G": _screen_row(soup=0.1230, last10=0.1270),
    }
    deltas = cv2.screening_deltas(rows)
    gates = {kind: cv2.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    assert not gates["F"]["passed"] and gates["R"]["passed"] and gates["G"]["passed"]
    selection = cv2.select_winner(rows, gates)
    assert selection["winner"] == "G"
    assert selection["reason"] == "SCREENING_SOUP_MAE"
    # inside the tolerance the frozen F > R > G order decides
    rows["F"] = _screen_row(soup=0.1250, last10=0.1270)
    rows["R"] = _screen_row(soup=0.1255, last10=0.1272)
    rows["G"] = _screen_row(soup=0.1248, last10=0.1268)
    deltas = cv2.screening_deltas(rows)
    gates = {kind: cv2.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    assert all(gates[kind]["passed"] for kind in gates)
    selection = cv2.select_winner(rows, gates)
    assert selection["winner"] == "F"
    assert selection["reason"] == "TIE_BREAK_F_R_G"
    # nobody passes -> local widening is not supported
    rows["F"] = _screen_row(soup=0.1290, last10=0.1300)
    rows["R"] = _screen_row(soup=0.1292, last10=0.1310)
    rows["G"] = _screen_row(soup=0.1289, last10=0.1305)
    deltas = cv2.screening_deltas(rows)
    gates = {kind: cv2.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    selection = cv2.select_winner(rows, gates)
    assert selection["winner"] is None
    assert selection["reason"] == "LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED"


def test_full_interpretation_bands() -> None:
    assert cv2.full_interpretation(0.1282)["band"] == "SHORT_SCREEN_SIGNAL_DID_NOT_TRANSFER"
    assert cv2.full_interpretation(0.1271)["band"] == "SHORT_SCREEN_SIGNAL_DID_NOT_TRANSFER"
    assert cv2.full_interpretation(0.1270)["band"] == "FULL_CAPACITY_GAIN_NOT_ESTABLISHED"
    assert cv2.full_interpretation(0.1260)["band"] == "FULL_CAPACITY_GAIN_NOT_ESTABLISHED"
    assert cv2.full_interpretation(0.1250)["band"] == "CAPACITY_DIRECTION_SUPPORTED_SINGLE_SEED"
    assert cv2.full_interpretation(0.1200)["band"] == "NEW_PERFORMANCE_BAND_SINGLE_SEED"
    assert cv2.full_interpretation(0.1100)["band"] == "MAJOR_CAPACITY_BOTTLENECK_IDENTIFIED"
    payload = cv2.full_interpretation(0.1240)
    assert payload["delta_vs_cap_base"] == pytest.approx(0.1240 - cv2.CAP_BASE_SOUP_MAE, abs=1e-12)
    assert payload["delta_vs_final_clean_sparse"] == pytest.approx(
        0.1240 - cv2.FINAL_CLEAN_SPARSE_SOUP_MAE, abs=1e-12
    )


# ---------------------------------------------------------------------------
# guards and fingerprints
# ---------------------------------------------------------------------------


def test_cpu_only_guard() -> None:
    cv2.cpu_only_guard(torch.device("cpu"))
    with pytest.raises(RuntimeError):
        cv2.cpu_only_guard(torch.device("cuda"))


def test_official_test_blocker() -> None:
    cv2.official_test_blocker({"official_test_loaded": False})
    with pytest.raises(RuntimeError):
        cv2.official_test_blocker({"official_test_loaded": True})


def test_architecture_fingerprints_are_stable_and_distinct() -> None:
    subspace = _subspace(seed=20)
    dictionary = _dictionary()
    rows = {}
    for kind in cv2.KINDS:
        model = cv2.build_v2_model(dictionary, 0, subspace, kind)
        rows[kind] = cv2.architecture_fingerprint(kind, model)
        assert rows[kind]["params"] == runner.V1_TOTAL_PARAMS[kind]
    assert rows["F"]["added_params"] == runner.V1_ADDED_PARAMS["F"]
    assert rows["R"]["added_params"] == runner.V1_ADDED_PARAMS["R"]
    assert rows["G"]["added_params"] == runner.V1_ADDED_PARAMS["G"]
    digests = {kind: rows[kind]["shape_fingerprint_sha256"] for kind in cv2.KINDS}
    assert len(set(digests.values())) == len(digests)
    assert rows["F"]["constants"]["fusion_residual_scale"] == cv2.FUSION_RESIDUAL_SCALE
    assert rows["R"]["constants"]["relation_residual_scale"] == cv2.RELATION_RESIDUAL_SCALE
    # the fingerprint is deterministic
    model = cv2.build_v2_model(dictionary, 0, subspace, "F")
    assert (
        cv2.architecture_fingerprint("F", model)["shape_fingerprint_sha256"]
        == digests["F"]
    )


# ---------------------------------------------------------------------------
# warm loop smoke test
# ---------------------------------------------------------------------------


def test_train_warm_smoke_on_tiny_subset() -> None:
    subspace = _frozen_subspace()
    state = _soup_state()
    train = p1run.load_split("train", subset=8)
    valid = p1run.load_split("valid", subset=8)
    if not train or not valid:  # pragma: no cover
        pytest.skip("ZINC encoded data not available")
    model = cv2.build_v2_model(_dictionary(), 0, subspace, "G")
    cv2.load_capacity_warm_state(model, state)
    optimizer, report = cv2.build_warm_optimizer(model)
    assert report["new_parameter_count"] > 0
    payload = cv2.train_warm(
        tag="SMOKE-V2",
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
        last_k=2,
        log=False,
    )
    assert payload["epochs_run"] == 2
    assert len(payload["curve"]) == 2
    for row in payload["curve"]:
        for key in (
            "train_mae",
            "valid_mae",
            "train_rec",
            "train_rec_term",
            "train_total_loss",
            "grad_norm_base",
            "grad_norm_new",
            "update_norm_base",
            "update_norm_new",
            "seconds",
        ):
            assert key in row, key
        assert np.isfinite(float(row[key]))
        assert float(row["grad_norm_base"]) > 0.0
        assert float(row["grad_norm_new"]) > 0.0
        assert float(row["update_norm_new"]) > 0.0
    assert payload["optimizer_groups"]["base_lr"] == cv2.BASE_LR
    assert payload["optimizer_groups"]["new_lr"] == cv2.NEW_LR
    assert payload["branch_grad_max"] > 0.0
    assert payload["branch_update_max"] > 0.0
    assert payload["branch_grad_finite"] and payload["branch_update_finite"]
    assert len(payload["soup_members"]) == 2
    assert np.isfinite(payload["soup_valid_mae"])
    assert payload["official_test_loaded"] is False


# ---------------------------------------------------------------------------
# runner plumbing
# ---------------------------------------------------------------------------


def test_runner_import_and_arm_layout() -> None:
    assert set(runner.ARM_DIRS) == set(cv2.KINDS)
    assert runner.ARM_DIRS["M0"].name == "m0"
    assert runner.ARM_DIRS["G"].name == "readout"
    assert runner.PROTOCOL_VERSION == cv2.PROTOCOL_VERSION
    assert tuple(runner.PROBES) == (
        "P1_phi_row_shuffle",
        "P2_node_correspondence",
        "P3_edge_correspondence",
        "P4_relation",
    )
    assert runner.RESULTS_DIR.name == "e2e_dictenv_capacity_localization_v2"
    assert runner.V1_ADDED_PARAMS == {"F": 29568, "R": 34400, "G": 30336}


def test_runner_csv_round_trip(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "table.csv"
    runner._write_csv(path, rows, ["a", "b", "missing"])
    read = runner._read_csv_rows(path)
    assert [row["a"] for row in read] == ["1", "2"]
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
    callback = runner._full_early_stop_callback(reference, [])
    stopped: int | None = None
    for epoch in range(1, 61):
        keep, payload = callback(epoch, model, None, [{"epoch": epoch, "train_mae": 0.1, "valid_mae": 0.5}])
        if not keep:
            stopped = epoch
            assert payload["reason"] == "matched_cssd_curve_hopeless"
            break
    assert stopped == 40
    callback = runner._full_early_stop_callback(reference, [])
    for epoch in range(1, 121):
        keep, _payload = callback(epoch, model, None, [{"epoch": epoch, "train_mae": 0.1, "valid_mae": 0.1}])
        assert keep


def test_runner_parameter_budget_matches_v1() -> None:
    subspace = _subspace(seed=21)
    budget = runner.parameter_budget_v2(subspace, _dictionary())
    assert budget["cap_base_params"] == 97727
    assert budget["all_candidates_match_v1"]
    for kind in ("F", "R", "G"):
        row = budget["rows"][kind]
        assert row["matches_v1_added_params"] and row["matches_v1_total_params"]
        assert row["total_params"] == runner.V1_TOTAL_PARAMS[kind]
        assert row["added_params_vs_cap_base"] == runner.V1_ADDED_PARAMS[kind]
    assert budget["added_params_ratio_ok"]
    assert budget["official_test_loaded"] is False


def test_screen_arm_requires_calibration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "CALIBRATION_DIR", tmp_path / "calibration")
    with pytest.raises(RuntimeError):
        runner.stage_screen_arm("F")
