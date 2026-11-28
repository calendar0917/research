"""CPU invariants of the SGA-v1 round (no GPU, no training).

Run: ``PYTHONPATH=code uv run pytest -q tracks/structural-sufficiency/structural_gain_attribution_v1/tests``
Some tests need the local caches built by ``build_inputs`` / ``run_p0_oov`` /
``run_phaseC_zinc_prep``; they skip when a cache is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "code"))

import sga_common as K  # noqa: E402

K.add_prev_code_path()

import sga_chem as CHEM  # noqa: E402


# ---------------------------------------------------------------------------
# Phase 0.1 chemical decoding
# ---------------------------------------------------------------------------


def test_atomic_num_index_zero_based():
    """``possible_atomic_num_list[i]`` is the element with atomic number i+1."""
    for idx in range(0, 30):
        d = CHEM.decode_atom([idx, 0, 0, 5, 0, 0, 0, 0, 0])
        assert d["atomic_num"] == idx + 1
    d5 = CHEM.decode_atom([5, 0, 0, 5, 0, 0, 2, 0, 1])
    assert d5["symbol"] == "C", "index 5 is carbon, not boron (the MSDS-v0 display bug)"
    old = CHEM.predecessor_coarse_atom_label([5, 5, 0])
    assert old.startswith("B"), "the old decoder really said boron for a carbon atom"
    assert old == "B+5", "and it also read the charge category index 5 as a +5 charge"


def test_formal_charge_hybridization_bond_tables():
    d = CHEM.decode_atom([5, 0, 3, 5, 1, 0, 2, 1, 1])
    assert d["formal_charge"] == 0 and d["hybridization"] == "SP3"
    assert CHEM.decode_bond([0, 0, 0])["bond_type"] == "SINGLE"
    assert CHEM.decode_bond([3, 0, 1])["bond_type"] == "AROMATIC"
    assert CHEM.decode_bond([1, 1, 0])["stereo"] == "STEREOZ"


def test_coarse_label_uses_only_coarse_columns():
    assert CHEM.atom_label([5, 5, 0], "COARSE") == "C"
    assert CHEM.atom_label([7, 6, 1], "COARSE") == "Oar+1"


def test_describe_key_is_display_only_and_round_trips():
    key = (((5, 5, 0), (5, 5, 0)), ((0, 1, (0,)),))
    d = CHEM.describe_key(key, "COARSE")
    assert d["n_nodes"] == 2 and d["n_bonds"] == 1
    assert d["node_signature"] == "a:Cx2"
    assert d["bond_labels"] == ["SINGLE"]


# ---------------------------------------------------------------------------
# Phase A representations
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle():
    if not K.INPUT_BUNDLE.exists():
        pytest.skip("input bundle not built")
    with np.load(K.INPUT_BUNDLE) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def test_pres_is_exactly_one_over_sqrt_n(bundle):
    n_atoms = np.asarray(bundle["n_atoms"], dtype=np.float64)
    inv = 1.0 / np.sqrt(np.maximum(n_atoms, 1.0))
    data = np.asarray(bundle["PRES_data"], dtype=np.float64)
    rows = np.repeat(np.arange(n_atoms.size), np.diff(bundle["PRES_indptr"]))
    dev = np.abs(data / inv[rows] - 1.0)
    assert dev.max() < 1e-6
    assert np.all(data > 0)


def test_pres_zero_pattern_matches_cf_union(bundle):
    def csr(name):
        import scipy.sparse as sp

        indices = np.asarray(bundle[f"{name}_indices"], dtype=np.int64)
        return sp.csr_matrix(
            (np.asarray(bundle[f"{name}_data"], dtype=np.float32), indices,
             np.asarray(bundle[f"{name}_indptr"], dtype=np.int64)),
            shape=(int(bundle["n_atoms"].size), int(indices.max()) + 1))


    c = csr("C").tocsr()
    f = csr("F").tocsr()
    pres = csr("PRES").tocsr()
    import scipy.sparse as sp

    both = (sp.hstack([c, f], format="csr") > 0).astype(np.int8)
    assert np.array_equal(both.toarray(), (pres > 0).astype(np.int8).toarray())


def test_bundle_h_round_trips(bundle):
    import sga_historical as HIST

    n_bits = int(bundle["H_n_bits"])
    shape = tuple(int(v) for v in bundle["H_shape"])
    bits = np.unpackbits(bundle["H_bits_packed"], axis=1)[:, :n_bits].astype(np.float32)
    h = np.empty(shape, dtype=np.float32)
    h[:, :n_bits] = bits
    h[:, n_bits:] = np.asarray(bundle["H_desc"], dtype=np.float32)
    assert np.array_equal(h[:, :n_bits], (h[:, :n_bits] > 0).astype(np.float32))
    assert np.isfinite(h).all()


def test_mc_support_equals_binary_morgan(bundle):
    meta = json.loads(K.INPUT_BUNDLE_META.read_text())
    assert meta["morgan_binary_block"]["n_parse_failed"] == 0
    assert meta["mc_meta"]["support_identity_holds"] is True
    assert meta["mc_equals_fold_of_mu"]["identical"] is True


def test_wl_and_mu_are_train_topk(bundle):
    meta = json.loads(K.INPUT_BUNDLE_META.read_text())
    for key in ("mu_meta", "wl_meta"):
        assert meta[key]["k_selected"] == 4406
        assert meta[key]["train_df_min_selected"] >= 5
        assert meta[key]["selection"].startswith("Train document frequency desc")
    assert meta["wl_meta"]["n_selected_by_round"]["0"] > 0


def test_labels_and_split_sizes(bundle):
    y = np.asarray(bundle["y"])
    assert set(np.unique(y).tolist()) <= {0, 1}
    assert bundle["rows_train"].size == 32901
    assert bundle["rows_valid"].size == 4113
    assert bundle["rows_test"].size == 4113


# ---------------------------------------------------------------------------
# Phase B matched initialisation
# ---------------------------------------------------------------------------


def test_matched_mlp_arms_share_initialisation():
    torch = pytest.importorskip("torch")
    import run_phaseB_mlp as B

    m0, h0 = B.make_model(2258, 0)
    mw, hw = B.make_model(2258, 4406)
    mc, hc = B.make_model(2258, 4406)
    assert h0 == hw == hc, "shared tensors must be identical across arms"
    assert float(mw.w_q.weight.abs().sum()) == 0.0, "the extra branch must start at exactly zero"
    assert torch.equal(m0.w_h.weight, mw.w_h.weight)
    assert B.n_parameters(mc)["total"] - B.n_parameters(m0)["total"] == 4406 * 128


def test_mlp_matches_protocol_architecture():
    import run_phaseB_mlp as B

    assert B.HIDDEN == 128 and B.DROPOUT == 0.1 and B.BATCH == 256
    assert B.LR == 1e-3 and B.WEIGHT_DECAY == 1e-4 and B.CLIP == 5.0 and B.EPOCHS == 50


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def test_paired_bootstrap_of_identical_predictions_is_zero():
    import msds_stats as ST

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400).astype(np.float64)
    logit = rng.normal(size=400)
    groups = rng.integers(0, 40, size=400)
    out = ST.paired_scaffold_bootstrap(y, logit, logit, groups, n_boot=50)
    assert out["point"]["delta_roc_auc"] == 0.0
    assert out["delta_roc_auc"]["p2.5"] == 0.0 == out["delta_roc_auc"]["p97.5"]


def test_mae_bootstrap_sign_convention():
    import run_analysis_abc as A

    y = np.array([0.0, 1.0, 2.0, 3.0])
    good = y.copy()
    bad = y + 1.0
    out = A.paired_molecule_bootstrap_mae(y, good, bad, n_boot=64)
    assert out["mae_a"] == 0.0 and out["mae_b"] == 1.0
    assert out["delta_mae"] == pytest.approx(1.0), "positive = arm a has the lower error"
    assert out["bootstrap_delta_mae"]["p2.5"] > 0


# ---------------------------------------------------------------------------
# Phase 0.2 OOV partition
# ---------------------------------------------------------------------------


def test_oov_classes_partition_all_types():
    path = K.ANALYSIS / "phase0_oov.json"
    if not path.exists():
        pytest.skip("Phase 0.2 not run")
    d = json.loads(path.read_text())
    assert d["n_train_seen_types"] == (d["n_in_vocab_types"]
                                       + d["class_counts"]["train_seen_but_not_frozen"])
    assert d["n_full_types"] == d["n_in_vocab_types"] + d["class_counts"]["train_seen_but_not_frozen"] \
        + d["class_counts"]["train_never_seen"]
    assert d["class_counts"]["train_never_seen"] == d["n_train_never_seen_types"]
    assert sum(d["reason_counts"][r] for r in
               ("parent_coarse_not_retained", "below_min_document_frequency",
                "below_min_scaffold_frequency", "top_k_truncated_by_family_cap")
               ) == d["class_counts"]["train_seen_but_not_frozen"]


# ---------------------------------------------------------------------------
# Phase C ZINC
# ---------------------------------------------------------------------------


def test_zinc_typed_wl_is_deterministic():
    pytest.importorskip("torch")
    import sga_zinc as Z

    try:
        data = Z.load_split("valid")
    except FileNotFoundError:
        pytest.skip("ZINC processed data not available")
    g = data["graphs"][0]
    a = Z.typed_rooted_signatures(g)
    b = Z.typed_rooted_signatures(g)
    assert a == b
    assert all(len(round_sigs) == int(g["x"].size) for round_sigs in a)


def test_zinc_design_matches_its_meta():
    meta_path = K.CACHE / "zinc_design_meta.json"
    design = K.CACHE / "zinc_design.npz"
    if not meta_path.exists() or not design.exists():
        pytest.skip("ZINC design not built")
    meta = json.loads(meta_path.read_text())
    assert meta["Z_CF"]["full_to_coarse"]["conflicts"] == 0
    assert meta["Z_CF"]["full_to_coarse"]["types_without_retained_parent"] == 0
    with np.load(design) as z:
        assert z["rows_train"].size == 10000 and z["rows_valid"].size == 1000
        assert z["rows_test"].size == 1000
        assert int(z["C_indptr"].size - 1) == 12000
    assert K.sha256_file(design) == meta["design"]["sha256"]


def test_zinc_features_use_dataset_classes():
    meta_path = K.CACHE / "zinc_design_meta.json"
    if not meta_path.exists():
        pytest.skip("ZINC design not built")
    meta = json.loads(meta_path.read_text())
    assert meta["Z"]["node_classes"] == list(range(21))
    assert meta["Z"]["edge_classes"] == [1, 2, 3]
    assert meta["Z_CF"]["filters"]["scaffold_gate"].startswith("dropped")
