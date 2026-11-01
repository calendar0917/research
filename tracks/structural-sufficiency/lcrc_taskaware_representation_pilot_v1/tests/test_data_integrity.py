"""P0 correctness tests: label-freedom, train-only normalisation, split discipline.

A future representation change must keep every one of these passing; they are
the executable form of the three historical PHYSCHEM6 failures (row-index
misalignment, double standardisation, wrong ``rho`` initialisation on a wider
input).
"""

from __future__ import annotations

import numpy as np
import pytest

import ta_common as C


# ---------------------------------------------------------------------------
# 4./5. candidate features are deterministic, label-free, and split-safe
# ---------------------------------------------------------------------------


def test_scaffold_folds_are_deterministic_and_do_not_cross_folds():
    import ta_cv

    published = ta_cv.bbbp_published_rows()
    import mb2_source as S

    src = S.load_dataset("BBBP", labels="train_valid")
    rows = list(published["rows"]["train"])[:200]
    y = np.asarray(src["y"]["train"], dtype=np.float64).reshape(-1)[:200]
    scaffolds = ta_cv.murcko_scaffolds(src["smiles"], rows)
    a = ta_cv.build_folds(rows, y, scaffolds)
    b = ta_cv.build_folds(rows, y, scaffolds)
    assert a["assignment"] == b["assignment"]
    for key, fold in a["assignment"].items():
        members = [r for r in rows if (scaffolds[r] or f"__acyclic__/{r}") == key]
        present = {f for f in range(3) if any(r in set(a["folds"][f]["rows"]) for r in members)}
        assert present == {fold}
    union: list[int] = []
    for f in range(3):
        union.extend(a["folds"][f]["rows"])
    assert sorted(union) == sorted(rows)


def test_frozen_folds_reproduce_from_the_frozen_rows():
    import ta_cv

    entry = ta_cv.frozen_folds()
    for fold in range(C.BBBP_CV_FOLDS):
        train_rows, valid_rows = ta_cv.rows_for_fold(fold)
        assert C.row_sha(valid_rows) == entry["folds"][str(fold)]["rows_sha256"]
        assert not (set(train_rows) & set(valid_rows))
        assert len(train_rows) + len(valid_rows) == entry["n_train_rows"]


def test_structure_caches_carry_no_labels():
    import ta_data as TD

    rows = {"train": [3, 17, 101, 250], "valid": [7, 42]}
    cache, info = TD.build_or_load_cache("BBBP", rows["train"], tag="test_label_free", split_label="x")
    assert cache["meta"]["labels_present"] is False
    for key in cache:
        assert key not in ("y", "target", "label", "labels")
    assert "rows" in cache and "smiles" in cache


def test_recomputed_descriptors_are_deterministic_and_label_blind():
    import mb2_source as S
    import ta_coverage as COV
    from rdkit import Chem

    smiles = S.all_smiles("FreeSolv")[5]
    mol_a = Chem.MolFromSmiles(smiles)
    mol_b = Chem.MolFromSmiles(smiles)
    fa = COV._molecule_facts(mol_a)
    fb = COV._molecule_facts(mol_b)
    assert fa == fb
    # the fact extractor never receives a label; it only takes a mol object
    import inspect

    signature = inspect.signature(COV._molecule_facts)
    assert list(signature.parameters) == ["mol"]


# ---------------------------------------------------------------------------
# 7. train-only normalisation, applied exactly once
# ---------------------------------------------------------------------------


def test_normaliser_is_fitted_on_the_given_train_rows_only():
    import lcrc_data as PREP_DATA
    import ta_data as TD

    train_rows = [3, 17, 101, 250, 400]
    other_rows = [7, 42, 77, 99, 123]
    cache_a, _ = TD.build_or_load_cache("BBBP", train_rows, tag="test_norm_a", split_label="a")
    cache_b, _ = TD.build_or_load_cache("BBBP", other_rows, tag="test_norm_b", split_label="b")
    norm_a = PREP_DATA.fit_norm(cache_a, TD.D.frozen_spec(), source="test_a")
    norm_b = PREP_DATA.fit_norm(cache_b, TD.D.frozen_spec(), source="test_b")
    assert norm_a["valid_or_test_used"] is False
    assert norm_a["fit_split"] == "BBBP_a"
    assert PREP_DATA.norm_fingerprint(norm_a) != PREP_DATA.norm_fingerprint(norm_b)
    # a constant feature must not be divided by the legacy 1e-6 epsilon
    const = [i for i, v in enumerate(np.asarray(norm_a["scalar_scale"])) if v == 1.0]
    assert const == list(norm_a["scalar_constant_indices"])


def test_scalars_are_standardised_exactly_once_inside_the_model():
    import lcrc_data as PREP_DATA
    import mb2_data as D
    import mb2_model as M
    import torch

    import ta_data as TD

    rows = [3, 17, 101, 250]
    cache, _ = TD.build_or_load_cache("BBBP", rows, tag="test_norm_once", split_label="once")
    spec = D.frozen_spec()
    norm = PREP_DATA.fit_norm(cache, spec, source="test_once")
    model, _meta = M.build_lcrc(spec, norm, 0, torch.device("cpu"), dropout=0.0)
    model.eval()
    captured = {}

    def hook(_module, args):
        captured["scal"] = args[3].detach().clone()

    handle = model.local.register_forward_pre_hook(hook)
    batch = PREP_DATA.collate(cache, list(range(len(rows))))
    with torch.no_grad():
        model(batch)
    handle.remove()
    raw = torch.as_tensor(np.asarray(batch["atom_scalar"]), dtype=torch.float32)
    mean = torch.as_tensor(np.asarray(norm["scalar_mean"]), dtype=torch.float32)
    scale = torch.as_tensor(np.asarray(norm["scalar_scale"]), dtype=torch.float32)
    expected_once = (raw - mean) / scale
    assert torch.allclose(captured["scal"], expected_once, atol=1e-6)
    double = ((raw - mean) / scale - mean) / scale
    assert not torch.allclose(captured["scal"], double, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. input-width / parameter-count audit hooks for a future representation change
# ---------------------------------------------------------------------------


def test_frozen_model_parameter_identity_and_width_guards():
    import lcrc_data as PREP_DATA
    import mb2_data as D
    import mb2_model as M
    import torch

    import ta_data as TD

    cache, _ = TD.build_or_load_cache("BBBP", [3, 17, 101, 250], tag="test_params", split_label="p")
    spec = D.frozen_spec()
    norm = PREP_DATA.fit_norm(cache, spec, source="test_params")
    single, meta1 = M.build_lcrc(spec, norm, 0, torch.device("cpu"), dropout=0.0, n_tasks=1)
    multi, meta12 = M.build_lcrc(spec, norm, 0, torch.device("cpu"), dropout=0.0, n_tasks=12)
    assert meta1["parameters"] == 247_106
    assert meta12["parameters"] == 247_106 + 1430
    # the shared encoder must have identical initialisation at the same seed
    assert meta1["shared_encoder_state_hash"] == meta12["shared_encoder_state_hash"]
    # rho input width is derived from the spec, so a wider input cannot silently
    # reuse the old initialisation: the constructor validates the normaliser shape
    bad_norm = dict(norm)
    bad_norm["global_mean"] = np.zeros(spec.global_dim + 1, dtype=np.float32)
    bad_norm["global_scale"] = np.ones(spec.global_dim + 1, dtype=np.float32)
    with pytest.raises(ValueError):
        M.build_lcrc(spec, bad_norm, 0, torch.device("cpu"), dropout=0.0)
    # the expected rho width must be recomputed from the spec, not hard-coded
    widths = meta1["input_widths"]
    assert widths["rho_input"] == spec.global_dim + 2 * int(meta1["config"]["width"]) + 1
    assert widths["atom_feat"] == spec.atom_feat_dim


# ---------------------------------------------------------------------------
# losses: differ on missing-label patterns, correct gradients
# ---------------------------------------------------------------------------


def test_equal_task_loss_differs_and_gradients_are_correct():
    import ta_losses as L

    rows = L.toy_probe()
    by_case = {r["case"]: r for r in rows}
    assert by_case["sparse_two_tasks"]["losses_differ"] is True
    assert by_case["empty_third_task"]["losses_differ"] is True
    assert by_case["empty_third_task"]["n_tasks_with_an_observed_cell"] == 2
    assert by_case["only_first_task"]["losses_differ"] is False
    for row in rows:
        assert row["grad_masked_cell_matches_analytic"] is True
        assert row["grad_equal_task_matches_analytic"] is True


def test_hiv_pos_weight_is_train_only_and_capped():
    import mb2_source as S
    import ta_losses as L

    y = np.asarray(S.load_dataset("ogbg-molhiv", labels="train_valid")["y"]["train"], dtype=np.float64).reshape(-1)
    info = L.sqrt_pos_weight(y, cap=5.0)
    assert info["pos_weight"] <= 5.0
    assert info["pos_weight"] == pytest.approx(min(5.0, float(np.sqrt(info["n_negative"] / info["n_positive"]))))
    assert info["source"] == "official Train split labels only"


# ---------------------------------------------------------------------------
# checkpoint recheck tolerance must follow the resolution of the statistic
# ---------------------------------------------------------------------------


def test_recheck_tolerance_scales_with_auc_resolution():
    import ta_train

    y = np.zeros((545, 1), dtype=np.float64)
    y[:448, 0] = 1.0
    tol, basis = ta_train._recheck_tolerance("binary", y, None)
    step = 1.0 / (448 * 97)
    assert tol >= 1.5 * step - 1e-15
    assert "97" in basis and "448" in basis

    # a dense fold goes back to the flat float tolerance
    y_big = np.zeros((10000, 1), dtype=np.float64)
    y_big[:5000, 0] = 1.0
    tol_big, _ = ta_train._recheck_tolerance("binary", y_big, None)
    assert tol_big == pytest.approx(max(1e-5, 1.5 / (5000 * 5000)))

    # multitask: driven by the sparsest definable task
    y_mt = np.full((200, 3), np.nan)
    y_mt[:80, 0] = 1.0
    y_mt[80:150, 0] = 0.0
    y_mt[:20, 1] = 1.0
    y_mt[20:30, 1] = 0.0
    tol_mt, basis_mt = ta_train._recheck_tolerance("multitask", y_mt, None)
    assert tol_mt == pytest.approx(max(1e-5, 1.5 / (20 * 10)))
    assert "over 2 tasks" in basis_mt

    # regression keeps the fixed float tolerance
    tol_reg, _ = ta_train._recheck_tolerance("regression", np.zeros((10, 1)), None)
    assert tol_reg == pytest.approx(1e-5)
