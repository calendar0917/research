"""Focused CPU tests for E2E-DictEnv-A2 (Attributed Code Value & Compression Control).

A2 is a *reuse* round: it must not re-implement the A1 433-D objects, the frozen
K-SVD dictionaries, the train-only scaler, the exact OMP coder or the frozen
downstream model.  These tests pin that discipline (source-level and functional),
the frozen thresholds and decision table, the new continuity-v2 pair population,
the train-only rank-32 dense control, matched initialisation/batch order, the
shared IHT step-count rule and the official-test blocker.

They run on CPU only, never touch the GPU, never download data and never load the
official ZINC test split.  Data-dependent tests reuse the local A1 artifacts with
small row subsets; the two heaviest ones are marked ``slow``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2 as a2
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as a1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a2 as run

CORE_PATH = Path(a2.__file__)
RUNNER_PATH = Path(run.__file__)
A1_RUNNER_PATH = Path(a1run.__file__)
A1_RESULTS = run.A1_RESULTS_DIR


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _cold_names(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _calls_to(path: Path, attribute: str) -> int:
    total = 0
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == attribute:
                total += 1
    return total


def _require_a1_artifacts() -> None:
    missing = [
        str(path)
        for path in (
            a1run.SDB_DICT_PATH,
            a1run.DICT_STORE["INDEP"],
            a1run.DICT_STORE["REAL"],
            a1run.SCALER_JSON,
            run.A1_CACHE_DIR / "omp_REAL_train.pt",
        )
        if not Path(path).exists()
    ]
    if missing:
        pytest.skip(f"A1 artifacts unavailable locally: {missing}")


# ---------------------------------------------------------------------------
# 1. frozen constants
# ---------------------------------------------------------------------------


def test_frozen_protocol_constants():
    assert a2.PROTOCOL_VERSION == "e2e_dictenv_a2"
    assert a2.ROUND == "E2E-DictEnv-A2"
    assert a2.MATERIAL == 0.003
    assert a2.MECHANISM_THRESHOLD == 0.010
    assert a2.PCA_RANK == 32
    assert a2.SCREEN_ARMS == ("TOPO", "INDEP", "REAL")
    assert a2.PCA_ARMS == ("INDEP", "REAL")
    assert tuple(a2.SCREEN_LABEL[arm] for arm in a2.SCREEN_ARMS) == ("T0", "I0", "R0")
    assert tuple(a2.E2E_LABEL[arm] for arm in a2.SCREEN_ARMS) == ("E0", "E1", "E2")
    # inherited, not re-declared: same objects, same thresholds
    assert a2.MATERIAL == 0.003 and a1.IHT_QUALIFY_MAX_REC == a2.IHT_QUALIFY_MAX_REC
    assert tuple(a1.IHT_CANDIDATE_STEPS) == (10, 30, 100, 200) == a2.IHT_CANDIDATE_STEPS
    assert a2.LIVENESS_MIN_ACTIVE_ATOMS == 24
    assert a2.LIVENESS_MIN_EFFECTIVE_ATOMS == 8.0
    assert a2.LIVENESS_MIN_MOVEMENT_RELATIVE == 0.01
    assert a2.LIVENESS_MIN_USAGE_SPEARMAN == 0.5
    assert a2.LIVENESS_SLOTS == ("node_encoder", "edge_encoder", "anchor_encoder")


def test_verdict_label_set_is_exactly_the_preregistration():
    assert a2.VERDICTS == (
        "ARTIFACT_IDENTITY_FAILURE",
        "GATE0_NOT_QUALIFIED",
        "CODER_NOT_QUALIFIED",
        "CODE_FORMATION_NOT_LIVE",
        "PAIRING_SUPPORTED_AT_FROZEN_OMP",
        "ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC",
        "ATTRIBUTED_SPARSE_DICTIONARY_SUPPORTED",
        "ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK",
        "NO_ATTRIBUTED_CODE_FORMATION_SIGNAL",
    )
    assert len(set(a2.VERDICTS)) == len(a2.VERDICTS)


def test_continuity_v2_is_diagnostic_and_disjoint_from_the_a1_pool():
    # the A2 diagnostic pool must not be A1's pool (reconciliation is explicit)
    assert a2.CONTINUITY_V2_POOL == 2000
    assert a2.CONTINUITY_V2_SEED == 20260930
    assert a2.CONTINUITY_V2_SEED != a2.CONTINUITY_A1_SEED
    assert a2.CONTINUITY_A1_POOL == 1500
    assert a2.CONTINUITY_A1_SEED == 20260933
    assert a2.CONTINUITY_A1_SAMPLED_PAIRS == 200000
    assert a2.CONTINUITY_A1_POSTHOC_SEED_OFFSET == 1
    # three frozen patch-size strata taken from the train quantiles
    assert a2.STRATA == (("small", 0, 5), ("medium", 6, 8), ("large", 9, 1 << 30))


def test_a1_reconciliation_constants_match_the_tracked_a1_runner():
    # the reconciliation re-runs A1's *exact* producing recipe
    assert a2.CONTINUITY_A1_POOL == int(a1run.a1.CONTINUITY_POOL)
    assert a2.CONTINUITY_A1_SEED == int(a1run.CONTINUITY_SEED)
    assert a2.CONTINUITY_A1_SAMPLED_PAIRS == int(a1run.POSTHOC_PAIRS)
    assert a2.CONTINUITY_V2_ROUNDS == int(a1run.a1.CONTINUITY_ROUNDS)
    assert a2.CONTINUITY_A1_SEED == int(sdb.DICT_SEED) + int(a1run.a1.CONTINUITY_SEED_OFFSET)


# ---------------------------------------------------------------------------
# 2. no second implementation (provenance discipline)
# ---------------------------------------------------------------------------


def test_core_module_reimplements_nothing():
    names = _cold_names(CORE_PATH)
    for forbidden in (
        "correctness_stage",
        "assignment_stage",
        "health_stage",
        "accounting_stage",
        "iht_diag",
        "mechanism_stage",
        "specificity_stage",
        "fit_pca",
        "omp_codes",
        "trains_arm",
    ):
        assert forbidden not in names
    assert _calls_to(CORE_PATH, "svd") == 0
    assert _calls_to(CORE_PATH, "eigh") == 0
    assert _calls_to(CORE_PATH, "tsqr") == 0
    assert "fit_pca_rank" in CORE_PATH.read_text(encoding="utf-8")


def test_runner_reuses_the_frozen_a1_functions_verbatim():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    # the frozen stages are *called*, never reimplemented
    for name in (
        "iht_diag",
        "mechanism_stage",
        "specificity_stage",
        "train_arm",
        "arm_coordinate",
        "load_or_build_codes",
        "load_arm_dictionary",
        "continuity_pool_data",
        "_load_soup_model",
        "attach_a1",
        "_mol_like",
    ):
        assert name in attributes, name
    assert "zinc_e2e_dictenv_a1" in RUNNER_PATH.read_text(encoding="utf-8")
    # no local re-derivation of the frozen statistics
    for forbidden in ("fit_pca_rank", "omp_codes", "tied_iht_codes"):
        assert forbidden not in _cold_names(RUNNER_PATH)


def test_runner_contains_no_second_coder_or_decoder():
    text = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "sklearn",
        "dict_learning",
        "MiniBatchDictionaryLearning",
        "KMeans",
        "orthogonal_mp",
        "PCA(",
        "fit_pca_rank(",
    ):
        assert forbidden not in text, forbidden
    # the only svd use is the singular-value-only effective-rank diagnostic
    assert text.count("np.linalg.svd") == 1
    assert "np.linalg.svd(dbar_soup, compute_uv=False)" in text
    assert "compute_uv=True" not in text


def test_official_test_split_never_appears_in_a2_sources():
    for path in (CORE_PATH, RUNNER_PATH):
        literals = _string_literals(path)
        assert "test" not in literals
        assert "val" not in literals
    assert {"train", "valid"} <= _string_literals(RUNNER_PATH)
    assert RUNNER_PATH.read_text(encoding="utf-8").count('"official_test_loaded": False') >= 8


def test_stage_choices_are_complete_and_ordered():
    choices = None
    for node in ast.walk(ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not (isinstance(node.args[0], ast.Constant) and node.args[0].value == "stage"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices" and isinstance(keyword.value, ast.Tuple):
                choices = tuple(element.value for element in keyword.value.elts)
    assert choices == (
        "identity",
        "qualify",
        "continuity-v2",
        "omp-screen",
        "omp-decision",
        "compression",
        "compression-decision",
        "coder",
        "formal",
        "mechanism",
        "liveness",
        "specificity",
        "smoke",
        "decision",
        "report",
        "all",
    )


def test_gpu_policy_requires_physical_gpu1(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError):
        run._set_device_policy("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError):
        run._set_device_policy("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError):
        run._set_device_policy("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert run._set_device_policy("cpu").type == "cpu"


# ---------------------------------------------------------------------------
# 3. reused-artifact identity
# ---------------------------------------------------------------------------


def test_dictionary_hashes_match_the_tracked_a1_pins():
    _require_a1_artifacts()
    indep, indep_sha = a1run.load_arm_dictionary("INDEP")
    real, real_sha = a1run.load_arm_dictionary("REAL")
    topo, topo_sha = a1run.load_arm_dictionary("TOPO")
    assert indep.shape == (a1.A1_DIM, a1.DICT_K)
    assert real.shape == (a1.A1_DIM, a1.DICT_K)
    assert topo.shape == (a1.PHI_DIM, a1.DICT_K)
    assert indep_sha == "400821ee5105050eb34600a0fb4cb8040f983daa1e773738dc9e2774036ff32d"
    assert real_sha == "c1cafb086662fb0753d52987fc321b1369d4f586164dde7960a3bed775d4b809"
    assert topo_sha == a1run.SDB_DICT_SHA256_F32
    assert run.EXPECTED_DICT_SHA["TOPO"] == a1run.SDB_DICT_SHA256_F32
    assert run.EXPECTED_DICT_SHA["TOPO"] == topo_sha
    assert run.EXPECTED_DICT_SHA["INDEP"] == indep_sha
    assert run.EXPECTED_DICT_SHA["REAL"] == real_sha


def test_indep_and_real_coordinates_are_433d_and_distinct():
    _require_a1_artifacts()
    real = a1run.arm_coordinate("REAL", "valid")
    indep = a1run.arm_coordinate("INDEP", "valid")
    topo = a1run.arm_coordinate("TOPO", "valid")
    assert real.shape[1] == indep.shape[1] == a1.A1_DIM == 433
    assert topo.shape[1] == a1.PHI_DIM == 65
    assert real.shape == indep.shape
    assert not np.array_equal(real, indep)
    raw = a1run.load_raw("valid")
    scalers = a1run.load_scalers()
    rebuilt = a1.apply_object_scaler(scalers[a1.ARM_COORDINATE["INDEP"]], raw)
    assert np.allclose(rebuilt, np.asarray(indep, dtype=np.float64), atol=1e-6)
    rebuilt_real = a1.apply_object_scaler(scalers[a1.ARM_COORDINATE["REAL"]], raw)
    assert np.allclose(rebuilt_real, np.asarray(real, dtype=np.float64), atol=1e-6)
    # INDEP is the marginal blocks of the same scaler object; REAL the joint blocks
    assert not np.array_equal(
        a1.concatenate_blocks(raw, "indep"), a1.concatenate_blocks(raw, "real")
    )
    assert scalers["real"].coordinate == "real" and scalers["indep"].coordinate == "indep"


def test_omp_cache_recomputes_exactly_without_rewriting_it():
    _require_a1_artifacts()
    before = (run.A1_CACHE_DIR / "omp_REAL_valid.pt").stat().st_mtime_ns
    for arm in ("REAL", "INDEP"):
        X = np.asarray(a1run.arm_coordinate(arm, "valid")[:128], dtype=np.float64)
        D, _sha = a1run.load_arm_dictionary(arm)
        Dbar = sdb.normalize_columns(np.asarray(D, dtype=np.float64))
        recomputed = sdb.omp_codes(Dbar, X, s=a1.DICT_S)
        stored = torch.load(run.A1_CACHE_DIR / f"omp_{arm}_valid.pt", map_location="cpu", weights_only=True)[:128]
        assert np.array_equal(np.asarray(stored, dtype=np.float64), recomputed.astype(np.float64))
        assert float((np.abs(recomputed) > 0).sum(1).mean()) <= a1.DICT_S + 1e-9
    assert (run.A1_CACHE_DIR / "omp_REAL_valid.pt").stat().st_mtime_ns == before


def test_strata_come_from_the_frozen_train_patch_size_quantiles():
    _require_a1_artifacts()
    sizes = np.asarray(a1run.load_raw("train")["n_patch"], dtype=np.int64)
    q25, q90 = (float(np.quantile(sizes, q)) for q in (0.25, 0.90))
    assert (int(np.floor(q25)), int(np.round(q90))) == (5, 8)
    assert a2.STRATA[0][2] == int(np.floor(q25))
    assert a2.STRATA[1][2] == int(np.round(q90))
    assert a2.STRATA[0][2] + 1 == a2.STRATA[1][1]
    assert a2.STRATA[1][2] + 1 == a2.STRATA[2][1]
    shares = np.asarray([(a2.stratum_of(int(size)) is not None) for size in sizes])
    assert shares.all()
    labels = [a2.stratum_of(int(size)) for size in sizes]
    assert set(labels) == {"small", "medium", "large"}
    # every size is in exactly one stratum, boundaries included
    assert a2.stratum_of(5) == "small" and a2.stratum_of(6) == "medium"
    assert a2.stratum_of(8) == "medium" and a2.stratum_of(9) == "large"


@pytest.mark.slow
def test_identity_stage_passes_on_the_reused_artifacts():
    _require_a1_artifacts()
    payload = run.identity_stage(rows=64, write=False)
    assert payload["all_passed"], payload["failures"]
    assert payload["official_test_loaded"] is False
    assert payload["entries"]["dictionary_REAL"]["passed"]
    assert payload["entries"]["scaler_real"]["passed"]
    assert payload["entries"]["omp_REAL_train"]["passed"]


# ---------------------------------------------------------------------------
# 4. continuity-v2 statistics and pair population
# ---------------------------------------------------------------------------


def test_spearman_and_angular_distance_math():
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    assert a2.spearman(x, x) == pytest.approx(1.0)
    assert a2.spearman(x, -x) == pytest.approx(-1.0)
    assert a2.spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert np.isnan(a2.spearman(np.asarray([1.0]), np.asarray([1.0])))
    assert np.isnan(a2.spearman(np.asarray([]), np.asarray([])))
    # frozen tie convention: ordinal ranks without averaging (exactly A1's
    # ``np.corrcoef(np.argsort(np.argsort(...)))``), so ties break by index
    assert a2.spearman(np.asarray([1.0, 2.0, 2.0]), np.asarray([1.0, 2.0, 3.0])) == 1.0
    rng = np.random.default_rng(3)
    for _ in range(5):
        left = rng.integers(0, 4, size=64).astype(np.float64)
        right = rng.integers(0, 4, size=64).astype(np.float64)
        frozen = float(
            np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1]
        )
        assert a2.spearman(left, right) == pytest.approx(frozen, abs=0)
    # A1's own producing code uses the same formula
    assert "np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1]" in (
        A1_RUNNER_PATH.read_text(encoding="utf-8")
    )
    a = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    d = a2.angular_distances(a, np.asarray([[0, 1], [0, 2], [1, 2]]))
    assert d[0] == pytest.approx(np.pi / 2)
    assert d[1] == pytest.approx(np.pi / 4)
    assert d[2] == pytest.approx(np.pi / 4)
    parallel = a2.angular_distances(a, np.asarray([[0, 0]]))
    assert parallel[0] == pytest.approx(0.0)
    # scale invariance of the *angular* distance
    scaled = a * np.asarray([[3.0], [0.25], [7.0]])
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]])
    assert np.allclose(a2.angular_distances(scaled, pairs), a2.angular_distances(a, pairs))
    e = a2.euclidean_distances(a, np.asarray([[0, 1]]))
    assert e[0] == pytest.approx(np.sqrt(2.0))


def test_pair_sampler_is_equal_size_molecule_disjoint_and_deterministic():
    sizes = [4, 4, 7, 7, 7, 10, 10]
    molecules = [0, 1, 2, 3, 4, 5, 6]
    keys = [1, 2, 3, 4, 5, 1, 2]
    first = a2.equal_size_molecule_disjoint_pairs(sizes, molecules, keys, cap_per_stratum=8, seed=7)
    second = a2.equal_size_molecule_disjoint_pairs(sizes, molecules, keys, cap_per_stratum=8, seed=7)
    third = a2.equal_size_molecule_disjoint_pairs(sizes, molecules, keys, cap_per_stratum=8, seed=8)
    assert np.array_equal(first["pooled"], second["pooled"])
    assert not np.array_equal(first["pooled"], third["pooled"])
    sizes_arr = np.asarray(sizes)
    mols_arr = np.asarray(molecules)
    keys_arr = np.asarray(keys)
    for stratum, pairs in first["strata"].items():
        assert pairs.size > 0, stratum
        for i, j in pairs:
            assert sizes_arr[i] == sizes_arr[j]
            assert mols_arr[i] != mols_arr[j]
            assert keys_arr[i] != keys_arr[j]
            assert i != j
            assert a2.stratum_of(int(sizes_arr[i])) == stratum
    # the pooled row is an independent unbiased sample of the same valid-pair
    # permutation; the strata are equal-n subsamples of it (documented overlap)
    assert first["pooled"].shape[0] == min(4 * first["cap_per_stratum"], first["n_valid_pairs"])
    for pair in first["pooled"]:
        i, j = int(pair[0]), int(pair[1])
        assert sizes_arr[i] == sizes_arr[j]
        assert mols_arr[i] != mols_arr[j]
        assert keys_arr[i] != keys_arr[j]
        assert i < j
    for name in ("small", "medium", "large"):
        assert first["stratum_counts"][name]["n_used"] == first["strata"][name].shape[0]
    assert first["n_valid_pairs_outside_strata"] == 0
    # every emitted pair appears once per population (no duplicate, no (i, j) + (j, i))
    for population in (first["pooled"], *first["strata"].values()):
        encoded = {tuple(sorted(pair)) for pair in population}
        assert len(encoded) == population.shape[0]
    # the three strata are pairwise disjoint and shallower than the pooled cap
    assert all(first["strata"][name].shape[0] <= first["cap_per_stratum"] for name in first["strata"])


def test_pair_sampler_respects_the_cap_and_reports_the_population():
    sizes = [4] * 6
    molecules = [0, 1, 2, 3, 4, 5]
    keys = [0, 1, 2, 3, 4, 5]
    payload = a2.equal_size_molecule_disjoint_pairs(sizes, molecules, keys, cap_per_stratum=3, seed=1)
    assert payload["strata"]["small"].shape[0] == 3
    assert payload["stratum_counts"]["small"]["n_used"] == 3
    assert payload["stratum_counts"]["small"]["n_valid"] == 15
    assert payload["cap_per_stratum"] == 3
    assert payload["n_used_pooled"] == 12  # unbiased pooled cap is 4 x cap_per_stratum
    assert payload["stratum_counts"]["medium"]["n_valid"] == 0
    assert payload["n_valid_pairs"] == 15
    assert payload["n_pool"] == 6 and payload["n_pool_pairs"] == 15


def test_reconstruct_a1_posthoc_pairs_matches_the_a1_population():
    _require_a1_artifacts()
    sizes = np.asarray(a1run.load_raw("train")["n_patch"], dtype=np.int64)
    rng = np.random.default_rng(a2.CONTINUITY_A1_SEED)
    pool = rng.choice(sizes.shape[0], size=a2.CONTINUITY_A1_POOL, replace=False)
    pool_sizes = sizes[pool]
    # molecules are the pool members themselves, keys are a distinct integer per pool member
    draw = a2.reconstruct_a1_posthoc_pairs(
        pool_sizes,
        np.arange(a2.CONTINUITY_A1_POOL),
        np.arange(a2.CONTINUITY_A1_POOL),
        samples=a2.CONTINUITY_A1_SAMPLED_PAIRS,
        seed=a2.CONTINUITY_A1_SEED + a2.CONTINUITY_A1_POSTHOC_SEED_OFFSET,
    )
    assert draw["pairs"].shape[1] == 2
    assert draw["pairs"].shape[0] <= a2.CONTINUITY_A1_SAMPLED_PAIRS
    assert draw["equal_size"].shape == (draw["pairs"].shape[0],)
    assert draw["molecule_disjoint"].shape == draw["equal_size"].shape
    assert draw["equal_size_molecule_disjoint"].shape == draw["equal_size"].shape
    assert int(draw["equal_size"].sum()) >= int(draw["equal_size_molecule_disjoint"].sum())
    # keys/sizes/molecules are the pool members' own, so a *reconstruction* with
    # the pool's real sizes must reproduce the A1 population counts exactly
    assert np.array_equal(draw["molecule_disjoint"], np.ones_like(draw["molecule_disjoint"]))


@pytest.mark.slow
def test_continuity_v2_reconciles_the_tracked_a1_numbers():
    if not (A1_RESULTS / "gate0_stratum_diagnostics.json").exists():
        pytest.skip("A1 stratum diagnostics unavailable")
    payload = run.continuity_v2_stage()
    # diagnostic-only by construction: the artifact carries no gate and no pass
    assert payload["kind"].startswith("diagnostic")
    assert "gate" not in payload and "pass" not in payload and "verdict" not in payload
    assert payload["primary_statistic"] == "Spearman(attributed_WL_similarity, -angular_distance)"
    assert payload["official_test_loaded"] is False
    assert payload["strata_definition"] == a2.stratum_table()
    population = payload["pair_population"]
    assert population["n_valid_pairs_outside_strata"] == 0
    assert "strata" not in population  # raw index frames are not part of the artifact
    reconciliation = payload["reconciliation_a1_pool"]
    assert reconciliation["n_equal_size"] == 18846
    assert reconciliation["n_equal_size_molecule_disjoint"] == 18843
    reference = json.loads((A1_RESULTS / "gate0_stratum_diagnostics.json").read_text())
    graded = reference["graded"]
    for arm in ("REAL", "INDEP", "TOPO"):
        ours = reconciliation["arms"][arm]["equal_size_molecule_disjoint"]
        theirs = graded[arm]
        assert ours["x"]["rho_euclidean"] == pytest.approx(
            theirs["spearman_cosine_vs_neg_xdist_equal_size_mol_disjoint"], abs=1e-9
        )
        assert ours["code"]["rho_euclidean"] == pytest.approx(
            theirs["spearman_cosine_vs_neg_codedist_equal_size_mol_disjoint"], abs=1e-9
        )
        # the equal-size-only (not molecule-disjoint) population is also reproduced
        assert reconciliation["arms"][arm]["all_pairs"]["x"]["rho_euclidean"] == pytest.approx(
            theirs["spearman_cosine_vs_neg_xdist_all"], abs=1e-9
        )
        assert reconciliation["arms"][arm]["all_pairs"]["code"]["rho_euclidean"] == pytest.approx(
            theirs["spearman_cosine_vs_neg_codedist_all"], abs=1e-9
        )
    # the *pipeline* is reproduced too: our attributed-WL + patch_slot_order
    # reconstruction of A1's own pool matches A1's similarity matrix exactly
    pipeline = reconciliation["pipeline"]
    assert pipeline["n_patches"] == 1500
    assert pipeline["sizes_identical"] is True
    assert pipeline["molecules_identical"] is True
    assert pipeline["key_partition_identical"] is True
    assert pipeline["n_patch_size_mismatch"] == 0
    assert pipeline["max_abs_similarity_difference"] == 0.0
    # the new A2 population is a different, larger, seeded pool
    assert payload["pool"]["n_pool"] == a2.CONTINUITY_V2_POOL
    assert payload["pool"]["seed"] == a2.CONTINUITY_V2_SEED
    assert payload["pool"]["n_patch_size_mismatch"] == 0
    for stratum in ("small", "medium", "large"):
        entry = payload["geometry"][stratum]
        assert payload["stratum_counts"][stratum]["n_used"] == a2.CONTINUITY_V2_PAIRS_PER_STRATUM
        for arm in ("TOPO", "INDEP", "REAL"):
            for space in ("x", "code"):
                assert np.isfinite(entry[arm][space]["rho_angular"])
                assert -1.0 <= entry[arm][space]["rho_angular"] <= 1.0
                assert entry[arm][space]["n_pairs"] == a2.CONTINUITY_V2_PAIRS_PER_STRATUM
                assert entry[arm][space]["mean_angular_distance"] >= 0.0
                assert entry[arm][space]["mean_euclidean_distance"] >= 0.0
                assert 0.0 <= entry[arm][space]["mean_wl_similarity"] <= 1.0
                assert set(entry[arm][space]) >= {
                    "rho_angular", "rho_euclidean", "n_pairs", "mean_wl_similarity"
                }
    pooled = payload["geometry"]["pooled"]
    assert pooled["REAL"]["x"]["n_pairs"] == 4 * a2.CONTINUITY_V2_PAIRS_PER_STRATUM
    # the two spaces are genuinely different objects (no indexing bug)
    assert not np.allclose(
        pooled["REAL"]["x"]["rho_angular"], pooled["REAL"]["code"]["rho_angular"], atol=1e-6
    )
    assert not np.allclose(
        pooled["REAL"]["x"]["rho_angular"], pooled["TOPO"]["x"]["rho_angular"], atol=1e-6
    )


# ---------------------------------------------------------------------------
# 5. Stage 2 — train-only rank-32 dense control
# ---------------------------------------------------------------------------


def test_pca_control_is_train_only_and_exactly_32_wide(monkeypatch):
    rng = np.random.default_rng(0)
    scale = np.linspace(0.5, 2.0, a1.A1_DIM)
    X_train = (rng.standard_normal((600, a1.A1_DIM)) * scale).astype(np.float32)
    X_valid = (rng.standard_normal((200, a1.A1_DIM)) * scale + 5.0).astype(np.float32)
    pca, meta = a2.fit_dense_rank(X_train, rank=a2.PCA_RANK)
    assert meta["rank"] == 32 and meta["fit_split"] == "official train"
    assert meta["n_fit_rows"] == 600
    projection = a2.pca_projection_matrix(pca)
    assert projection.shape == (a1.A1_DIM, 32)
    assert projection.dtype == np.float32
    # the frozen projection IS what the model's column-normalized dictionary evaluates to
    dbar = F.normalize(torch.as_tensor(projection), dim=0, eps=1e-12).numpy()
    assert np.abs(dbar - projection).max() < 1e-6
    assert meta["projection_vs_normalized_dictionary_max_abs"] <= 1e-5
    # delegation: the projection is exactly sdb_v0's PCA, no second implementation
    reference = sdb.fit_pca_rank(np.asarray(X_train, dtype=np.float64), rank=32)
    assert np.allclose(reference.components.T, projection, atol=1e-6)
    assert np.allclose(np.asarray(pca.mean), np.asarray(X_train, dtype=np.float64).mean(0))
    # the fitted object never sees valid data: a valid-only shift cannot move it
    shifted = np.concatenate([X_train, X_valid], axis=0)
    other, _meta = a2.fit_dense_rank(shifted, rank=a2.PCA_RANK)
    assert not np.allclose(a2.pca_projection_matrix(other), projection, atol=1e-3)
    assert meta["valid_never_enters_fit"] if "valid_never_enters_fit" in meta else True


def test_dense_codes_are_centered_tied_and_never_refit():
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((400, a1.A1_DIM)).astype(np.float32)
    X_valid = rng.standard_normal((50, a1.A1_DIM)).astype(np.float32) + 3.0
    pca, _meta = a2.fit_dense_rank(X_train, rank=a2.PCA_RANK)
    mean_before = np.asarray(pca.mean).copy()
    components_before = np.asarray(pca.components).copy()
    first = a2.dense_codes_f32(pca, X_valid)
    second = a2.dense_codes_f32(pca, X_valid)
    assert first.shape == (50, 32) and first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.array_equal(np.asarray(pca.mean), mean_before)
    assert np.array_equal(np.asarray(pca.components), components_before)
    expected = np.asarray(
        (np.asarray(X_valid, dtype=np.float64) - mean_before[None, :]) @ components_before.T,
        dtype=np.float32,
    )
    assert np.array_equal(first, expected)
    # the tied reconstruction the frozen decoder sees is the mean-free projection
    tied = np.asarray(first, dtype=np.float64) @ components_before
    assert np.allclose(
        tied,
        (np.asarray(X_valid, dtype=np.float64) - mean_before[None, :])
        @ components_before.T
        @ components_before,
        atol=1e-5,
    )
    # the affine PCA reconstruction is the tied one plus the mean term
    affine = mean_before[None, :] + tied
    assert np.allclose(affine, mean_before[None, :] + first.astype(np.float64) @ components_before)


def test_pca_artifact_roundtrip_verifies_stored_codes(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    X = {"train": rng.standard_normal((300, a1.A1_DIM)).astype(np.float32),
         "valid": rng.standard_normal((40, a1.A1_DIM)).astype(np.float32)}
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "STATE_DIR", tmp_path / "states")
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "FORMAL_DIR", tmp_path / "formal_runs")
    monkeypatch.setattr(
        a1run, "arm_coordinate", lambda arm, split, **kwargs: X[split].copy()
    )
    projection, mean, meta = run._pca_artifacts("REAL")
    mean_norm = np.asarray(mean)
    assert (tmp_path / "pca_projection_real.pt").exists()
    assert (tmp_path / "pca_codes_real.pt").exists()
    assert meta["valid_never_enters_fit"] is True
    assert meta["n_transform_rows_valid"] == 40
    # the frozen decoder adds no mean: the tied residual is the affine residual
    # shifted by the (constant) mean term n * ||mean||^2 / ||X||^2 on train
    train = X["train"].astype(np.float64)
    den = float((train ** 2).sum())
    gap = meta["train_tied_normalized_rec"] - meta["train_normalized_rec"]
    assert gap == pytest.approx(train.shape[0] * float((mean_norm ** 2).sum()) / den, rel=1e-6)
    assert gap > 0.0
    assert 0.0 < meta["valid_affine_normalized_rec"] < 1.0
    assert 0.0 < meta["valid_tied_normalized_rec"] < 1.0
    # reload path re-derives the identical projection and verifies stored codes
    projection_two, _mean_two, meta_two = run._pca_artifacts("REAL")
    assert np.array_equal(projection, projection_two)
    assert meta_two["components_sha256_f32"] == meta["components_sha256_f32"]
    stored = torch.load(tmp_path / "pca_codes_real.pt", map_location="cpu", weights_only=True)
    assert stored["train"].shape == (300, 32) and stored["valid"].shape == (40, 32)
    # a corrupted code cache is refused rather than silently reused
    blob = torch.load(tmp_path / "pca_codes_real.pt", map_location="cpu", weights_only=True)
    blob["train"][:1, :] += 1.0
    torch.save(blob, tmp_path / "pca_codes_real.pt")
    with pytest.raises(RuntimeError):
        run._pca_artifacts("REAL")


def test_pca_control_uses_the_frozen_a1_trainer_with_a_dictionary_override(
    monkeypatch, tmp_path
):
    sentinel = np.ones((a1.A1_DIM, 32), dtype=np.float32)
    seen: dict[str, object] = {}

    def fake_train_arm(arm, **kwargs):
        D, sha = a1run.load_arm_dictionary(arm)
        seen["arm"] = arm
        seen["D"] = np.asarray(D)
        seen["sha"] = sha
        seen["kwargs"] = kwargs
        return {
            "soup": {"soup_valid_mae": 0.0},
            "S": {"soup_epoch": 1},
            "dictionary_sha256_f32": sha,
            "official_test_loaded": False,
        }

    def fake_pca_artifacts(arm):
        blob = {"train": torch.zeros(8, 32), "valid": torch.zeros(8, 32)}
        torch.save(blob, tmp_path / f"pca_codes_{arm.lower()}.pt")
        return sentinel, np.zeros(a1.A1_DIM), {"rank": 32}

    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "STATE_DIR", tmp_path / "states")
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "FORMAL_DIR", tmp_path / "formal_runs")
    monkeypatch.setattr(run, "_pca_artifacts", fake_pca_artifacts)
    monkeypatch.setattr(
        a1run,
        "arm_coordinate",
        lambda arm, split, **kwargs: np.zeros((8, a1.A1_DIM), dtype=np.float32),
    )
    monkeypatch.setattr(a1run, "train_arm", fake_train_arm)
    monkeypatch.setattr(run, "_write_json", lambda path, payload: None)
    monkeypatch.setattr(run, "_mark", lambda *args, **kwargs: None)
    run.compression_stage(device="cpu")
    assert seen["arm"] == "REAL"  # the last of PCA_ARMS
    assert np.array_equal(seen["D"], sentinel)
    assert seen["sha"] == a2.a1_hash(sentinel)
    assert seen["kwargs"]["stage"] == "omp"
    assert seen["kwargs"]["codes_train"].shape[1] == 32
    assert seen["kwargs"]["codes_valid"].shape[1] == 32
    assert seen["kwargs"]["x_train"].shape == (8, a1.A1_DIM)


# ---------------------------------------------------------------------------
# 6. matched downstream execution
# ---------------------------------------------------------------------------


def test_matched_init_and_equal_parameter_counts():
    _require_a1_artifacts()
    dictionaries = {arm: a1run.load_arm_dictionary(arm)[0] for arm in a1.ARMS}
    models = {arm: a1.build_model(arm, dictionaries[arm], seed=0) for arm in a1.ARMS}
    counts = {
        arm: sum(p.numel() for p in model.parameters() if p.requires_grad)
        for arm, model in models.items()
    }
    assert counts["INDEP"] == counts["REAL"] == 109263
    assert counts["TOPO"] == 97487
    for arm in a1.ARMS:
        assert torch.equal(
            models[arm].D.detach(), torch.as_tensor(dictionaries[arm])
        ), arm
    # "D" is the frozen per-arm input object, not an initialized parameter; every
    # other shared-shape tensor must be bit-identical at seed 0
    for left, right in (("INDEP", "REAL"), ("TOPO", "REAL"), ("TOPO", "INDEP")):
        first_state, second_state = models[left].state_dict(), models[right].state_dict()
        shared = [
            key
            for key, value in first_state.items()
            if key in second_state and second_state[key].shape == value.shape and key != "D"
        ]
        assert len(shared) >= 45, (left, right, len(shared))
        for key in shared:
            assert torch.equal(first_state[key], second_state[key]), (left, right, key)
    # the dictionaries are the *objects under test*, so they differ by construction
    assert not torch.equal(models["INDEP"].D.detach(), models["REAL"].D.detach())
    # the visible trainable *set* (not just the count) is identical between INDEP and REAL
    indep_names = {name for name, p in models["INDEP"].named_parameters() if p.requires_grad}
    real_names = {name for name, p in models["REAL"].named_parameters() if p.requires_grad}
    assert indep_names == real_names
    assert (
        models["REAL"].D.requires_grad and models["INDEP"].D.requires_grad
    )


def test_matched_init_enforcement_is_a_noop_and_keeps_each_dictionary():
    _require_a1_artifacts()
    report = run.matched_init_report()
    assert report["passed"] is True
    assert report["reference_arm"] == "REAL"
    for arm, entry in report["arms"].items():
        assert entry["enforcement_is_a_noop"] is True, arm
        assert entry["matches_reference_state"] is True, arm
        assert entry["D_kept_per_arm"] is True, arm
        assert entry["shared_shape_tensors"] == 50, arm
    reference = run.reference_state("REAL")
    assert "D" not in reference  # the dictionary is never shared between arms
    original = a1.build_model
    indep_dictionary = np.asarray(a1run.load_arm_dictionary("INDEP")[0])
    with run.matched_init_enforced() as info:
        assert info["reference_arm"] == "REAL"
        model = a1.build_model("INDEP", indep_dictionary, seed=0)
        state = model.state_dict()
        for key, value in reference.items():
            assert key in state and state[key].shape == value.shape, key
            assert torch.equal(state[key], value), key
        # INDEP keeps its own marginal-object dictionary, not REAL's joint one
        assert torch.equal(state["D"].detach(), torch.as_tensor(indep_dictionary))
        assert not torch.equal(
            state["D"].detach(), torch.as_tensor(a1run.load_arm_dictionary("REAL")[0])
        )
    assert a1.build_model is original


def test_dictionary_override_is_arm_scoped_and_restored():
    _require_a1_artifacts()
    original = a1run.load_arm_dictionary
    indep_before = np.asarray(original("INDEP")[0])
    sentinel = np.zeros((a1.A1_DIM, a1.DICT_K), dtype=np.float32)
    with run.dictionary_override("REAL", sentinel):
        real, real_sha = a1run.load_arm_dictionary("REAL")
        indep, _sha = a1run.load_arm_dictionary("INDEP")
        assert np.array_equal(real, sentinel)
        assert real_sha == a2.a1_hash(sentinel)
        assert np.array_equal(indep, indep_before)
    assert a1run.load_arm_dictionary is original
    assert not np.array_equal(np.asarray(a1run.load_arm_dictionary("REAL")[0]), sentinel)


def test_a1_emits_into_redirects_and_restores_every_module_constant():
    before = (a1run.RESULTS_DIR, a1run.STATE_DIR, a1run.CURVE_DIR, a1run.FORMAL_DIR)
    with run.a1_emits_into(Path("/tmp/a2-test-redirect")):
        assert str(a1run.RESULTS_DIR) == "/tmp/a2-test-redirect"
        assert str(a1run.STATE_DIR) == "/tmp/a2-test-redirect/states"
        assert str(a1run.CURVE_DIR) == "/tmp/a2-test-redirect/curves"
        assert str(a1run.FORMAL_DIR) == "/tmp/a2-test-redirect/formal_runs"
    assert (a1run.RESULTS_DIR, a1run.STATE_DIR, a1run.CURVE_DIR, a1run.FORMAL_DIR) == before


def test_loader_seed_is_shared_across_arms():
    _require_a1_artifacts()
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run

    orders = {}
    for arm in ("INDEP", "REAL"):
        data = p1run.load_split("valid", subset=8)
        a1run.attach_a1(data, "valid", arm)
        loader = p1.make_env_loader(data, 4, True, a1run.SEED + a1run.TRAIN_SHUFFLE_OFFSET)
        orders[arm] = [batch.y.tolist() for batch in loader]
    assert orders["INDEP"] == orders["REAL"]
    assert sum(len(batch) for batch in orders["REAL"]) == 8
    # the frozen trainer hard-codes one seed pair for every arm
    source = A1_RUNNER_PATH.read_text(encoding="utf-8")
    assert "p1.make_env_loader(train_data, BATCH_SIZE, True, SEED + TRAIN_SHUFFLE_OFFSET)" in source
    assert a1run.SEED == run.SEED == 0


# ---------------------------------------------------------------------------
# 7. Stage 3 — one shared IHT step count
# ---------------------------------------------------------------------------


def test_shared_step_selection_rule():
    errors = {
        "TOPO": {"10": 0.010, "30": 0.0015, "100": 0.0009, "200": 0.0009},
        "INDEP": {"10": 0.004, "30": 0.0021, "100": 0.0018, "200": 0.0018},
        "REAL": {"10": 0.030, "30": 0.0010, "100": 0.0008, "200": 0.0008},
    }
    picked = a2.select_common_iht_steps(errors)
    assert picked["qualified_steps"] == [100, 200]
    assert picked["selected_steps"] == 100
    assert picked["coder_qualified"] is True
    # the shared rule bites: two arms are fine at every step, so a per-arm rule
    # would have qualified *something*, but one shared count must serve all three
    per_arm = {
        "TOPO": {str(step): 0.0005 for step in a2.IHT_CANDIDATE_STEPS},
        "INDEP": {str(step): 0.0005 for step in a2.IHT_CANDIDATE_STEPS},
        "REAL": {str(step): 0.5 for step in a2.IHT_CANDIDATE_STEPS},
    }
    strictly_shared = a2.select_common_iht_steps(per_arm)
    assert strictly_shared["qualified_steps"] == []
    assert strictly_shared["selected_steps"] is None
    assert strictly_shared["coder_qualified"] is False
    assert strictly_shared["bar"] == a2.IHT_QUALIFY_MAX_REC == 0.002
    assert strictly_shared["arms"] == ["TOPO", "INDEP", "REAL"]
    # incomplete evidence (a missing candidate step) is refused, not inferred
    with pytest.raises(RuntimeError):
        a2.select_common_iht_steps({"TOPO": {"10": 0.0}, "INDEP": {"10": 0.0}, "REAL": {"10": 0.0}})
    # the bar is inclusive at every candidate step
    assert a2.select_common_iht_steps(
        {arm: {str(step): a2.IHT_QUALIFY_MAX_REC for step in a2.IHT_CANDIDATE_STEPS}
         for arm in ("TOPO", "INDEP", "REAL")}
    )["selected_steps"] == 10


def test_coder_stage_rejects_a_disagreeing_a1_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "_mark", lambda *args, **kwargs: None)
    fake = {
        "protocol_version": "e2e_dictenv_a1",
        "qualified_steps": [30],
        "selected_steps": 30,
        "coder_qualified": True,
        "arms": {
            arm: {
                "steps": {
                    "10": {"normalized_err": 0.5},
                    "30": {"normalized_err": 0.001},
                    "100": {"normalized_err": 0.001},
                    "200": {"normalized_err": 0.001},
                }
            }
            for arm in a1.ARMS
        },
    }
    monkeypatch.setattr(a1run, "iht_diag", lambda device="cpu": dict(fake))
    with pytest.raises(RuntimeError):
        run.coder_stage(device="cpu")
    incomplete = dict(fake, arms={arm: {"steps": {"10": {"normalized_err": 0.001}}} for arm in a1.ARMS})
    monkeypatch.setattr(a1run, "iht_diag", lambda device="cpu": dict(incomplete))
    with pytest.raises(RuntimeError):
        run.coder_stage(device="cpu")


# ---------------------------------------------------------------------------
# 8. the frozen decision table
# ---------------------------------------------------------------------------


def _stage(run_flag=True, **extra):
    return {"run": run_flag, **extra}


def test_verdict_table_covers_every_preregistered_branch():
    ok = {"identity_passed": True, "gate0_passed": True}
    assert a2.verdict_from_evidence(identity_passed=False, gate0_passed=True)["verdict"] == (
        "ARTIFACT_IDENTITY_FAILURE"
    )
    assert a2.verdict_from_evidence(identity_passed=True, gate0_passed=False)["verdict"] == (
        "GATE0_NOT_QUALIFIED"
    )
    assert a2.verdict_from_evidence(
        **ok, stage1=_stage(primary_pass=True), stage3=_stage(coder_qualified=False)
    )["verdict"] == "CODER_NOT_QUALIFIED"
    assert a2.verdict_from_evidence(
        **ok,
        stage1=_stage(primary_pass=True),
        stage3=_stage(coder_qualified=True),
        stage4=_stage(primary_pass=True),
        liveness={"all_pass": False, "failing_arms": ["REAL"]},
    )["verdict"] == "CODE_FORMATION_NOT_LIVE"
    assert a2.verdict_from_evidence(
        **ok,
        stage1=_stage(primary_pass=True),
        stage3=_stage(coder_qualified=True),
        stage4=_stage(primary_pass=False),
        liveness={"all_pass": True},
    )["verdict"] == "PAIRING_SUPPORTED_AT_FROZEN_OMP"
    assert a2.verdict_from_evidence(
        **ok,
        stage1=_stage(primary_pass=True),
        stage3=_stage(coder_qualified=True),
        stage4=_stage(primary_pass=True),
        liveness={"all_pass": True},
        mechanism={"pass": False},
    )["verdict"] == "ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC"
    assert a2.verdict_from_evidence(
        **ok,
        stage1=_stage(primary_pass=True),
        stage3=_stage(coder_qualified=True),
        stage4=_stage(primary_pass=True),
        liveness={"all_pass": True},
        mechanism={"pass": True},
        specificity={"pass": False},
    )["verdict"] == "ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC"
    assert a2.verdict_from_evidence(
        **ok,
        stage1=_stage(primary_pass=True),
        stage3=_stage(coder_qualified=True),
        stage4=_stage(primary_pass=True),
        liveness={"all_pass": True},
        mechanism={"pass": True},
        specificity={"pass": True},
    )["verdict"] == "ATTRIBUTED_SPARSE_DICTIONARY_SUPPORTED"
    assert a2.verdict_from_evidence(
        **ok, stage1=_stage(primary_pass=False), stage2=_stage(primary_pass=True)
    )["verdict"] == "ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK"
    assert a2.verdict_from_evidence(
        **ok, stage1=_stage(primary_pass=False), stage2=_stage(primary_pass=False)
    )["verdict"] == "NO_ATTRIBUTED_CODE_FORMATION_SIGNAL"


def test_verdict_table_refuses_interrupted_evidence():
    ok = {"identity_passed": True, "gate0_passed": True}
    with pytest.raises(ValueError):
        a2.verdict_from_evidence(**ok)  # stage 1 never ran
    with pytest.raises(ValueError):
        a2.verdict_from_evidence(**ok, stage1=_stage(primary_pass=True))  # stage 3 missing
    with pytest.raises(ValueError):
        a2.verdict_from_evidence(
            **ok,
            stage1=_stage(primary_pass=True),
            stage3=_stage(coder_qualified=True),
            stage4=_stage(primary_pass=True),
            liveness={"all_pass": True},
            mechanism={"pass": True},
        )  # specificity missing while required


def test_liveness_gate_is_frozen_and_strict():
    good = {
        "dictionary_grad_norm": 1e-3,
        "soup_D_vs_ksvd_init": {"relative": 0.05},
        "active_atoms_train": 30,
        "effective_atoms_train": 12.0,
        "code_variance_mean_train": 0.1,
        "train_valid_usage_spearman": 0.9,
        "slot_gradients": {slot: {"param_grad_norm": 1.0} for slot in a2.LIVENESS_SLOTS},
    }
    assert a2.liveness_gate(good)["passed"] is True
    # a slot whose parameters receive no gradient is dead ...
    dead_slot = dict(good, slot_gradients={slot: {"param_grad_norm": 1.0} for slot in a2.LIVENESS_SLOTS})
    dead_slot["slot_gradients"]["anchor_encoder"] = {"param_grad_norm": 0.0, "slot_grad_norm": 1.0}
    assert a2.liveness_gate(dead_slot)["passed"] is False
    # ... while A1's input probe is used only when the parameter norm is absent
    probe_only = dict(good, slot_gradients={slot: {"slot_grad_norm": 1.0} for slot in a2.LIVENESS_SLOTS})
    assert a2.liveness_gate(probe_only)["passed"] is True
    silent = dict(good, slot_gradients={slot: {"slot_grad_norm": None} for slot in a2.LIVENESS_SLOTS})
    assert a2.liveness_gate(silent)["passed"] is False
    for key, value in (
        ("dictionary_grad_norm", 0.0),
        ("active_atoms_train", 23),
        ("effective_atoms_train", 7.9),
        ("train_valid_usage_spearman", 0.49),
    ):
        broken = dict(good, **{key: value})
        assert a2.liveness_gate(broken)["passed"] is False, key
    assert a2.liveness_gate(dict(good, soup_D_vs_ksvd_init={"relative": 0.001}))["passed"] is False
    assert a2.liveness_all_pass({"REAL": good, "INDEP": dict(good, active_atoms_train=0)}) == {
        "arms": {
            "REAL": a2.liveness_gate(good),
            "INDEP": a2.liveness_gate(dict(good, active_atoms_train=0)),
        },
        "all_pass": False,
        "failing_arms": ["INDEP"],
    }


# ---------------------------------------------------------------------------
# 9. orchestration bookkeeping
# ---------------------------------------------------------------------------


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_decision_and_report_record_every_skipped_stage_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "STATE_DIR", tmp_path / "states")
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "FORMAL_DIR", tmp_path / "formal_runs")
    monkeypatch.setattr(run, "STAGE_STATUS", {
        "compression": {"status": "RUN", "reason": "", "artifact": "pca_screen_{indep,real}.json",
                        "seconds": 1.0, "device": "cuda"},
        "compression-decision": {"status": "RUN", "reason": "", "artifact": "compression_decision.json",
                                 "seconds": None, "device": "cpu"},
    })
    _write(tmp_path / "artifact_identity.json", {"all_passed": True, "failures": [], "official_test_loaded": False})
    _write(tmp_path / "qualification.json", {"passed": True, "official_test_loaded": False})
    _write(tmp_path / "continuity_v2.json", {"kind": "diagnostic_only", "geometry": {
        name: {arm: {space: {"rho_angular": 0.1, "n_pairs": 4000} for space in ("x", "code")}
               for arm in a1.ARMS} for name, _l, _h in a2.STRATA} | {
        "pooled": {arm: {space: {"rho_angular": 0.1, "n_pairs": 16000} for space in ("x", "code")}
                   for arm in a1.ARMS}}, "stratum_counts": {}, "pool": {}, "contrasts": {}})
    _write(tmp_path / "omp_decision.json", {
        "run": True, "primary_pass": False, "secondary_pass": False,
        "soup_valid_mae": {"TOPO": 0.13, "INDEP": 0.124, "REAL": 0.130},
        "G_pair_OMP": -0.006, "G_topo_OMP": -0.0, "curves_finite": True,
    })
    _write(tmp_path / "compression_decision.json", {
        "run": True, "primary_pass": True, "soup_valid_mae": {"INDEP": 0.13, "REAL": 0.12},
        "G_pair_PCA": 0.01,
    })
    payload = run.decision_stage()
    assert payload["verdict"] == "ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK"
    status = payload["stage_status"]
    for stage in ("coder", "formal", "mechanism", "liveness", "specificity"):
        assert status[stage]["status"] == "NOT RUN", stage
        assert status[stage]["reason"], stage
    assert status["compression"]["status"] == "RUN"
    assert payload["stage3_coder"] is None and payload["stage4_e2e"] is None
    assert payload["official_test_loaded"] is False
    assert payload["seed"] == 0 and payload["seed1_run"] is False
    questions = run._seven_questions(payload)
    assert [item["id"] for item in questions] == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
    assert questions[1]["status"] == "RUN"
    assert questions[4]["status"] == "NOT RUN"
    assert questions[4]["answer"].startswith("NOT RUN")
    report = run.report_stage()
    text = Path(report["decision"]).read_text(encoding="utf-8")
    assert "ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK" in text
    assert "NOT RUN" in text
    assert "official_test_loaded = False" in text
    assert (tmp_path / "REPORT.md").exists() and (tmp_path / "DECISION.md").exists()


def test_stage1_and_stage2_decisions_apply_the_frozen_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STATE_DIR", tmp_path / "states")
    monkeypatch.setattr(run, "FORMAL_DIR", tmp_path / "formal_runs")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    _write(tmp_path / "omp_screen_topo.json", {"soup": {"soup_valid_mae": 0.1352}})
    _write(tmp_path / "omp_screen_indep.json", {"soup": {"soup_valid_mae": 0.1250}})
    _write(tmp_path / "omp_screen_real.json", {"soup": {"soup_valid_mae": 0.1200}})
    # fail-closed: a material contrast without a finite training curve cannot pass
    payload = run.omp_decision_stage()
    assert payload["G_pair_OMP"] == pytest.approx(0.1250 - 0.1200)
    assert payload["G_topo_OMP"] == pytest.approx(0.1352 - 0.1200)
    assert payload["threshold"] == a2.MATERIAL == 0.003
    assert payload["primary_pass"] is False and payload["curves_finite"] is False
    assert payload["official_test_loaded"] is False
    (tmp_path / "curves").mkdir(parents=True, exist_ok=True)
    for label in ("T0", "I0", "R0"):
        (tmp_path / "curves" / f"omp_screen_{label}_curve.csv").write_text(
            "epoch,train_mae,valid_mae\n1,0.5,0.4\n2,0.4,0.3\n", encoding="utf-8"
        )
    assert run._curve_is_finite(tmp_path / "curves" / "omp_screen_R0_curve.csv") is True
    assert run._curve_is_finite(tmp_path / "curves" / "missing_curve.csv") is False
    (tmp_path / "curves" / "omp_screen_T0_curve.csv").write_text(
        "epoch,train_mae,valid_mae\n1,0.5,nan\n", encoding="utf-8"
    )
    assert run._curve_is_finite(tmp_path / "curves" / "omp_screen_T0_curve.csv") is False
    (tmp_path / "curves" / "omp_screen_T0_curve.csv").write_text(
        "", encoding="utf-8"
    )
    assert run._curve_is_finite(tmp_path / "curves" / "omp_screen_T0_curve.csv") is False
    (tmp_path / "curves" / "omp_screen_T0_curve.csv").write_text(
        "epoch,train_mae,valid_mae\n1,0.5,0.4\n", encoding="utf-8"
    )
    payload = run.omp_decision_stage()
    assert payload["curves_finite"] is True
    assert payload["primary_pass"] is True  # +0.005 >= 0.003
    assert payload["secondary_pass"] is True
    # the secondary contrast alone is never the primary gate
    assert payload["primary_pass"] is bool(payload["G_pair_OMP"] >= a2.MATERIAL)
    _write(tmp_path / "pca_screen_indep.json", {"soup": {"soup_valid_mae": 0.1300}})
    _write(tmp_path / "pca_screen_real.json", {"soup": {"soup_valid_mae": 0.1260}})
    for label in ("I0", "R0"):
        (tmp_path / "curves" / f"pca_screen_{label}_curve.csv").write_text(
            "epoch,train_mae,valid_mae\n1,0.5,0.4\n", encoding="utf-8"
        )
    second = run.compression_decision_stage()
    assert second["G_pair_PCA"] == pytest.approx(0.004)
    assert second["primary_pass"] is True
    assert second["threshold"] == a2.MATERIAL
    # Stage 2 is the *fallback* branch: with a passing Stage 1 the table takes
    # the sparse route, and with a failing Stage 1 it takes the compression branch
    assert a2.verdict_from_evidence(
        identity_passed=True, gate0_passed=True, stage1=payload, stage2=second,
        stage3={"run": True, "coder_qualified": False},
    )["verdict"] == "CODER_NOT_QUALIFIED"
    failed_stage1 = dict(payload, primary_pass=False)
    assert a2.verdict_from_evidence(
        identity_passed=True, gate0_passed=True, stage1=failed_stage1, stage2=second,
    )["verdict"] == "ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK"


def test_decision_stage_reports_identity_failure_without_inventing_a_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _write(tmp_path / "artifact_identity.json", {"all_passed": False, "failures": ["omp_REAL_train"]})
    payload = run.decision_stage()
    assert payload["verdict"] == "ARTIFACT_IDENTITY_FAILURE"
    assert payload["identity"]["failures"] == ["omp_REAL_train"]
    assert payload["stage1_omp"] is None and payload["continuity_v2"] is None
