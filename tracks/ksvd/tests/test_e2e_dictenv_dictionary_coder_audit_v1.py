"""Focused CPU tests for E2E-DictEnv-Dictionary-Coder-Audit-v1.

Preregistration test surface (``e2e_dictenv_dictionary_coder_audit_v1`` section 10):

* IHT-10 reproduces the frozen existing coder exactly;
* IHT-30 / IHT-100 preserve the exact top-``s`` support;
* OMP honours the exact-``s`` contract and is no worse than IHT-10 in l2;
* the same-``D`` audit really uses one identical frozen ``Dbar``;
* support Jaccard / activation-frequency / effective-atom / Gini / entropy toy checks;
* dictionary matching under permutation and sign flip;
* mean direction in the dictionary's own input space;
* train-only PCA;
* structural profile train normalisation (valid uses the frozen train statistics);
* linear recoverability train-only fit;
* structural descriptors follow the phi65 provenance;
* CPU-only guard and official-test blocker;
* the IHT-30 gate is the frozen function (no threshold is re-tuned here);
* ``IHTStepModel`` with 10 steps is bit-identical to ``CleanMechModel``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_dictionary_coder_audit_v1 as dca
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_dictionary_coder_audit_v1 as runner


def _toy_dictionary(seed: int = 3, k: int = 8, features: int = 65) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    D = torch.randn(features, k, generator=generator)
    return np.asarray(D, dtype=np.float32)


def _toy_X(seed: int = 4, n: int = 16, features: int = 65) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    return np.asarray(torch.randn(n, features, generator=generator), dtype=np.float32)


def test_iht10_reproduces_existing_coder_exactly() -> None:
    D = _toy_dictionary()
    X = _toy_X()
    Dbar = dca.effective_dictionary(D)
    expected = v0.tied_iht_codes(
        torch.as_tensor(Dbar, dtype=torch.float32),
        torch.as_tensor(X, dtype=torch.float32),
        s=dca.SPARSITY,
        steps=10,
    )
    actual = dca.codes_for("iht10", X, Dbar)
    assert np.array_equal(actual, expected.detach().cpu().numpy().astype(np.float64))


def test_iht30_iht100_preserve_exact_top_s() -> None:
    D = _toy_dictionary()
    X = _toy_X()
    Dbar = dca.effective_dictionary(D)
    for coder in ("iht30", "iht100"):
        codes = dca.codes_for(coder, X, Dbar)
        support = np.abs(codes) > 0.0
        assert support.sum(axis=1).max() <= dca.SPARSITY
        assert np.all(support.sum(axis=1) == dca.SPARSITY)


def test_omp_exact_s_contract_and_l2_optimality() -> None:
    D = _toy_dictionary()
    X = _toy_X(seed=11)
    Dbar = dca.effective_dictionary(D)
    omp = dca.codes_for("omp", X, Dbar)
    iht = dca.codes_for("iht10", X, Dbar)
    assert np.all((np.abs(omp) > 0).sum(axis=1) == dca.SPARSITY)
    residual_omp = np.linalg.norm(X - omp @ Dbar.T, axis=1)
    residual_iht = np.linalg.norm(X - iht @ Dbar.T, axis=1)
    assert np.all(residual_omp <= residual_iht + 1e-8)


def test_same_d_audit_uses_one_identical_dictionary() -> None:
    D = _toy_dictionary(seed=5)
    X = _toy_X(seed=6)
    Dbar = dca.effective_dictionary(D)
    # effective_dictionary is exactly the model operator (column normalisation).
    expected = v0.normalized_dictionary(torch.as_tensor(D, dtype=torch.float32)).numpy().astype(np.float64)
    assert np.allclose(Dbar, expected, atol=1e-6, rtol=0.0)
    # rescaling the raw dictionary may not change the effective operator ...
    scaled = dca.effective_dictionary(D * 7.0)
    assert np.allclose(dca.codes_for("iht10", X, Dbar), dca.codes_for("iht10", X, scaled))
    # ... but a genuinely different direction must change the codes.
    other = Dbar.copy()
    other[:, 0] = np.roll(other[:, 0], 1)
    assert not np.allclose(dca.codes_for("iht10", X, Dbar), dca.codes_for("iht10", X, dca.effective_dictionary(other)))


def test_support_jaccard_toy_example() -> None:
    left = np.zeros((1, 4))
    right = np.zeros((1, 4))
    left[0, [0, 1]] = 1.0
    right[0, [1, 2]] = 1.0
    assert dca.support_jaccard(left, right)[0] == pytest.approx(1.0 / 3.0)
    assert dca.support_agreement(left, left)["exact_support_match_rate"] == 1.0


def test_activation_frequency_accounting() -> None:
    codes = np.zeros((4, 5))
    codes[0, [0, 1]] = 1.0
    codes[1, [1]] = 1.0
    codes[2, [1, 4]] = 1.0
    frequencies = dca.activation_frequency(codes)
    assert frequencies.tolist() == [0.25, 0.75, 0.0, 0.0, 0.25]
    concentration = dca.usage_concentration(frequencies)
    assert concentration["active_atoms"] == 3
    assert concentration["max_activation_rate"] == pytest.approx(0.75)


def test_effective_atom_count_formula() -> None:
    uniform = np.full(32, 0.5)
    assert dca.usage_concentration(uniform)["effective_atoms"] == pytest.approx(32.0)
    single = np.zeros(32)
    single[7] = 1.0
    assert dca.usage_concentration(single)["effective_atoms"] == pytest.approx(1.0)


def test_gini_and_entropy_toy_checks() -> None:
    uniform = np.ones(16) / 16.0
    assert dca.gini(uniform) == pytest.approx(0.0, abs=1e-12)
    assert dca.usage_concentration(uniform)["usage_entropy"] == pytest.approx(np.log(16.0))
    concentrated = np.zeros(16)
    concentrated[0] = 1.0
    assert dca.gini(concentrated) == pytest.approx(15.0 / 16.0)
    assert dca.usage_concentration(concentrated)["usage_entropy"] == pytest.approx(0.0)


def test_dictionary_matching_handles_permutation() -> None:
    D = dca.effective_dictionary(_toy_dictionary(seed=8))
    permutation = np.array([5, 0, 7, 2, 1, 6, 3, 4])
    permuted = D[:, permutation]
    result = dca.match_dictionaries(D, permuted)
    matched = {pair["atom_a"]: pair["atom_b"] for pair in result["pairs"]}
    inverse = np.argsort(permutation)
    assert matched == {int(i): int(inverse[i]) for i in range(len(permutation))}
    assert result["matched_abs_cosine"]["min"] == pytest.approx(1.0, abs=1e-6)
    assert result["n_above_095"] == len(permutation)


def test_dictionary_matching_handles_sign_flip() -> None:
    D = dca.effective_dictionary(_toy_dictionary(seed=9))
    flipped = D.copy()
    flipped[:, ::2] *= -1.0
    result = dca.match_dictionaries(D, flipped)
    assert result["matched_abs_cosine"]["min"] == pytest.approx(1.0, abs=1e-6)


def test_mean_direction_uses_the_passed_feature_space() -> None:
    X = np.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    mean = dca.mean_direction(X)
    assert mean.tolist() == [3.0, 6.0]
    Dbar = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    cosines = dca.atom_cosine_to_vector(Dbar, mean)
    assert cosines[0] == pytest.approx(3.0 / np.sqrt(45.0))
    # sign symmetry: negating either the atom or the direction is invariant
    assert dca.atom_cosine_to_vector(-Dbar, mean).tolist() == cosines.tolist()
    assert dca.atom_cosine_to_vector(Dbar, -mean).tolist() == cosines.tolist()


def test_pca_is_fit_on_train_only() -> None:
    generator = np.random.default_rng(0)
    train = generator.normal(size=(200, 4)) * np.asarray([5.0, 0.5, 0.2, 0.1])
    valid = generator.normal(size=(50, 4)) * np.asarray([0.1, 0.1, 0.1, 9.0])
    components, ratio, mean = dca.pca_basis(train, n_components=2)
    assert np.allclose(np.linalg.norm(components, axis=1), 1.0)
    assert np.abs(np.dot(components[0], np.asarray([1.0, 0.0, 0.0, 0.0]))) > 0.99
    assert mean.tolist() == np.asarray(train).mean(axis=0).tolist()
    valid_components, _valid_ratio, _valid_mean = dca.pca_basis(valid, n_components=2)
    assert np.abs(np.dot(valid_components[0], np.asarray([1.0, 0.0, 0.0, 0.0]))) < 0.5
    assert ratio[0] >= ratio[1]


def test_structural_descriptors_follow_phi_provenance() -> None:
    phi = np.zeros((2, dca.PHI_DIM))
    phi[0, 63] = np.log1p(7.0)
    phi[0, 64] = np.log1p(9.0)
    phi[0, 4] = np.log1p(3.0)
    phi[1, 13] = 0.25
    phi[1, 63] = np.log1p(4.0)
    phi[1, 41] = 0.5
    Z = dca.structural_descriptors(phi)
    index = {name: position for position, name in enumerate(dca.STRUCTURAL_DESCRIPTOR_NAMES)}
    assert Z[0, index["patch_nodes"]] == pytest.approx(7.0)
    assert Z[0, index["patch_edges"]] == pytest.approx(9.0)
    assert Z[0, index["root_induced_degree"]] == pytest.approx(3.0)
    assert Z[1, index["shell_pop1"]] == pytest.approx(1.0)
    assert Z[1, index["mean_edge_log1p_common_neighbours"]] == pytest.approx(0.5)


def test_structural_profile_uses_frozen_train_normalisation() -> None:
    n = 400
    width = len(dca.STRUCTURAL_DESCRIPTOR_NAMES)
    Z_train = np.zeros((n, width))
    Z_valid = np.zeros((n, width))
    Z_train[:, 0] = np.linspace(-1.0, 1.0, n) * 10.0
    Z_valid[:, 0] = np.linspace(-1.0, 1.0, n) * 1.0 + 5.0
    A = np.zeros((n, 4))
    A[n // 2 :, 0] = 1.0  # active only on the positive half of coordinate 0
    result = dca.atom_specialization_rows(
        seed=0, coder="toy", Z_train=Z_train, Z_valid=Z_valid, A_train=A, A_valid=A
    )
    row = result["rows"][0]
    std_train = Z_train[:, 0].std()
    expected_train = Z_train[n // 2 :, 0].mean() / std_train
    expected_valid = (Z_valid[n // 2 :, 0].mean() - Z_train[:, 0].mean()) / std_train
    profile_train = np.asarray(result["profiles"]["train_active"][0])
    profile_valid = np.asarray(result["profiles"]["valid_active"][0])
    assert profile_train[0] == pytest.approx(expected_train, rel=1e-9)
    # valid uses the frozen train mean/std, not its own statistics
    assert profile_valid[0] == pytest.approx(expected_valid, rel=1e-9)
    assert row["train_valid_profile_cosine"] == pytest.approx(1.0, abs=1e-6)
    # an atom that is never active has no profile
    assert np.isnan(result["rows"][1]["train_valid_profile_cosine"])


def test_linear_recoverability_fit_is_train_only() -> None:
    generator = np.random.default_rng(1)
    source_train = generator.normal(size=(300, 3))
    target_train = source_train @ np.asarray([[1.0, -2.0], [0.5, 0.25], [0.0, 3.0]]) + 0.5
    source_valid = generator.normal(size=(50, 3))
    target_valid = source_valid @ np.asarray([[1.0, -2.0], [0.5, 0.25], [0.0, 3.0]]) + 0.5
    metrics = dca.fit_linear_recoverability(source_train, target_train, source_valid, target_valid)
    assert metrics["valid_r2"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["linear_cka"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["valid_mean_cosine"]["mean"] == pytest.approx(1.0, abs=1e-6)
    # an independent train-only OLS reproduces the metric
    design = np.concatenate([source_train, np.ones((source_train.shape[0], 1))], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, target_train, rcond=None)
    prediction = np.concatenate([source_valid, np.ones((source_valid.shape[0], 1))], axis=1) @ coefficients
    manual = dca.recoverability_metrics(target_valid, prediction, target_train.mean(axis=0))
    assert manual["valid_r2"] == pytest.approx(metrics["valid_r2"])
    # a different valid target cannot change the fitted coefficients
    other = dca.fit_linear_recoverability(source_train, target_train, source_valid, target_valid * 0.0)
    assert other["valid_r2"] != pytest.approx(metrics["valid_r2"])


def test_cpu_only_guard() -> None:
    dca.cpu_only_guard(torch.device("cpu"))
    with pytest.raises(RuntimeError):
        dca.cpu_only_guard(torch.device("cuda"))


def test_official_test_blocker() -> None:
    dca.official_test_blocker({"official_test_loaded": False})
    with pytest.raises(RuntimeError):
        dca.official_test_blocker({"official_test_loaded": True})
    with pytest.raises(RuntimeError):
        dca.official_test_blocker({})
    # every payload builder carries the false blocker
    Z = np.random.default_rng(0).normal(size=(20, len(dca.STRUCTURAL_DESCRIPTOR_NAMES)))
    A = np.zeros((20, 4))
    A[:, 0] = 1.0
    specialization = dca.atom_specialization_rows(
        seed=0, coder="toy", Z_train=Z, Z_valid=Z, A_train=A, A_valid=A
    )
    assert specialization["official_test_loaded"] is False
    assert dca.match_dictionaries(np.eye(2), np.eye(2))["official_test_loaded"] is False
    assert (
        dca.fit_linear_recoverability(Z, Z, Z, Z)["official_test_loaded"] is False
    )
    # the runner has no official-test stage and only audits train/valid
    assert tuple(dca.SPLITS) == ("train", "valid")
    assert ("C6", 0) in runner.CHECKPOINTS and ("DENSE-TIED", 0) in runner.CHECKPOINTS
    assert runner.IHT30_EPOCHS == 320 and runner.IHT30_SANITY_EPOCHS == 20


def test_iht_step_model_matches_clean_mech_model_at_ten_steps() -> None:
    D = _toy_dictionary(seed=12)
    phi = torch.as_tensor(
        np.random.default_rng(3).normal(size=(6, dca.PHI_DIM)), dtype=torch.float32
    )
    torch.manual_seed(0)
    reference = cm.CleanMechModel(cm.H1_CONFIG, D)
    torch.manual_seed(0)
    candidate = runner.IHTStepModel(cm.H1_CONFIG, D, iht_steps=10)
    assert set(reference.state_dict().keys()) == set(candidate.state_dict().keys())
    for key, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        assert torch.equal(reference.code(phi), candidate.code(phi))
    torch.manual_seed(0)
    thirty = runner.IHTStepModel(cm.H1_CONFIG, D, iht_steps=30)
    thirty.load_state_dict(reference.state_dict())
    with torch.no_grad():
        codes = thirty.code(phi)
    assert torch.all((codes != 0).sum(dim=1) <= dca.SPARSITY)


def test_frozen_gate_uses_frozen_thresholds() -> None:
    good = {
        "iht10": {
            "frobenius": 1.0e-2,
            "support_jaccard_mean": 0.40,
            "effective_atoms": 12.0,
            "top5_share": 0.60,
            "max_activation_rate": 0.95,
            "top1_over_l1_mean": 0.40,
            "code_cosine_mean": 0.70,
        },
        "iht30": {
            "frobenius": 2.0e-3,
            "support_jaccard_mean": 0.60,
            "effective_atoms": 16.0,
            "top5_share": 0.45,
            "max_activation_rate": 0.70,
            "top1_over_l1_mean": 0.30,
            "code_cosine_mean": 0.90,
        },
        "omp": {"effective_atoms": 20.0, "top5_share": 0.30},
    }
    gate = dca.iht30_gate({0: good, 1: good, 2: good})
    assert gate["fired"] is True
    assert gate["supported_seeds"] == [0, 1, 2]
    assert gate["thresholds"]["recon_factor"] == dca.GATE_RECON_FACTOR
    weak = {seed: dict(good) for seed in (0, 1, 2)}
    for seed in weak:
        weak[seed]["iht30"] = dict(good["iht30"], frobenius=9.0e-3)
    assert dca.iht30_gate(weak)["fired"] is False


def test_audit_module_has_no_training_entry_point() -> None:
    """This round is an audit module; only the runner may build the IHT30 arm."""
    assert not hasattr(dca, "train")
    assert runner.IHT30_STEPS == 30
    assert audit.H1_LAMBDA == cm.H1_LAMBDA
