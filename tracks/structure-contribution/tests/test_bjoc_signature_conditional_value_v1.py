"""Fast correctness tests for cscl-bjoc-signature-conditional-value-v1.

No labels, no checkpoints, no GPU, no official test, no data files.  Covers the
conditioning math (molecule-capped residuals, shrinkage floors, context
residual), the stratified pseudo-control, Arm A/B/C scores and tie handling,
molecule-unit statistics, the frozen four-label decision rule, and the
label-discipline / freeze token scan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_signature_conditional_value_v1 as bcv  # noqa: E402
import bjoc_signature_transfer_v1 as bst  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/bjoc_signature_conditional_value_v1.yaml"

X = (0, (0, 0), (1, 1), 1, 1)
Y = (2, (1, 1), (1, 2), 0, 0)
KX = bst.sig_join(X)
KY = bst.sig_join(Y)


def _js(seed: int, row: int, keys, degree, erc, dev, ig, residual=None) -> bcv.JunctionSet:
    return bcv.junction_set_from_arrays(
        seed,
        row,
        [k if isinstance(k, str) else bst.sig_join(k) for k in keys],
        np.asarray(degree, dtype=np.int64),
        np.asarray(erc, dtype=np.int64),
        np.asarray(dev, dtype=np.float64),
        np.asarray(ig, dtype=np.float64),
        residual_override=residual,
    )


# ---------------------------------------------------------------------------
# frozen constants / config
# ---------------------------------------------------------------------------


def test_config_matches_library_constants():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == bcv.PROTOCOL_ID
    assert cfg["discovery"]["signature_floor_molecules"] == bcv.SIGNATURE_FLOOR_MOLECULES
    assert cfg["discovery"]["signature_shrink"] == bcv.SIGNATURE_SHRINK
    assert cfg["discovery"]["context_floor_molecules"] == bcv.CONTEXT_FLOOR_MOLECULES
    assert cfg["discovery"]["context_shrink"] == bcv.CONTEXT_SHRINK
    assert cfg["validation"]["top_k_primary"] == bcv.TOP_K_PRIMARY
    assert cfg["validation"]["labels"] == "NEVER READ"
    assert "NEVER" in cfg["validation"]["official_test"]
    assert cfg["pseudo_control"]["replicates"] == bcv.PSEUDO_REPLICATES
    assert cfg["pseudo_control"]["dev_bins"] == bcv.PSEUDO_DEV_BINS
    assert cfg["decision"]["recovery_fraction_min"] == bcv.RECOVERY_FRACTION_MIN
    assert cfg["decision"]["concentration_top1_share_max"] == bcv.CONCENTRATION_TOP1_SHARE_MAX
    for arm in ("A", "B", "C"):
        assert arm in cfg["arms"]


def test_percentile_matches_previous_round():
    for values in ([1.0], [3.0, 1.0, 2.0], [1.0, 1.0, 2.0, 2.0], [5.0, 4.0, 4.0, 1.0]):
        assert np.allclose(bcv.percentile(values), bst.average_rank_percentile(values))


# ---------------------------------------------------------------------------
# conditioning math
# ---------------------------------------------------------------------------


def test_shrunk_residual_floor_and_shrink():
    assert bcv.shrunk_residual(0.4, 19, 20.0, 20) == 0.0
    assert bcv.shrunk_residual(0.4, 20, 20.0, 20) == pytest.approx(0.5 * 0.4)
    assert bcv.shrunk_residual(-0.3, 80, 20.0, 20) == pytest.approx(0.8 * -0.3)


def test_molecule_cap_not_junction_weighted():
    """A molecule with 10 copies of a class must not dominate that class mean."""
    js_a = _js(0, 1, [KX] * 10, [2] * 10, [0] * 10, list(range(10)), list(range(10)), residual=[1.0] * 10)
    js_b = _js(0, 2, [KX], [2], [0], [0.0], [0.0], residual=[0.0])
    js_c = _js(0, 3, [KX], [2], [0], [0.0], [0.0], residual=[0.0])
    tables = bcv.build_residual_tables([js_a, js_b, js_c], signature_floor=2, context_floor=2)
    entry = tables["sig"][KX]
    assert entry["n_molecules"] == 3
    assert entry["R"] == pytest.approx(1.0 / 3.0)
    junction_weighted = 10.0 / 12.0
    assert abs(entry["R"] - junction_weighted) > 0.4


def test_context_residual_uses_shrunk_signature_bonus():
    """R_(tau,kappa) = molecule-capped mean of r - R'_tau."""
    j1 = _js(0, 1, [KX], [2], [0], [0.0], [0.0], residual=[0.8])
    j2 = _js(0, 2, [KX], [2], [0], [0.0], [0.0], residual=[0.6])
    j3 = _js(0, 3, [KX], [3], [1], [0.0], [0.0], residual=[0.2])
    tables = bcv.build_residual_tables([j1, j2, j3], signature_floor=2, context_floor=2, context_shrink=20.0)
    r_sig = (0.8 + 0.6 + 0.2) / 3.0
    r_sig_shrunk = bcv.shrunk_residual(r_sig, 3, 20.0, 2)
    assert tables["sig"][KX]["R"] == pytest.approx(r_sig)
    ctx_deg2 = bcv.context_key(KX, 2, 0)
    ctx_deg3 = bcv.context_key(KX, 3, 1)
    assert tables["ctx"][ctx_deg2]["R_raw"] == pytest.approx((0.8 + 0.6) / 2.0 - r_sig_shrunk)
    assert tables["ctx"][ctx_deg3]["R_raw"] == pytest.approx(0.2 - r_sig_shrunk)
    assert tables["ctx"][ctx_deg2]["n_molecules"] == 2
    # floors: one below-floor context unit falls back to zero
    tables2 = bcv.build_residual_tables([j1, j2, j3], signature_floor=2, context_floor=3)
    assert tables2["ctx"][ctx_deg2]["R_shrunk"] == 0.0
    assert tables2["ctx"][ctx_deg3]["R_shrunk"] == 0.0


def test_context_scope_includes_below_floor_signatures():
    """A below-floor signature still gets context units with R'_tau = 0."""
    js = [
        _js(0, 1, [KY], [2], [0], [0.0], [0.0], residual=[0.5]),
        _js(0, 2, [KY], [2], [0], [0.0], [0.0], residual=[0.3]),
    ]
    tables = bcv.build_residual_tables(js, signature_floor=20, context_floor=2)
    assert KY not in tables["sig"] or tables["sig"][KY]["R_shrunk"] == 0.0
    assert bcv.context_key(KY, 2, 0) in tables["ctx"]
    assert tables["ctx"][bcv.context_key(KY, 2, 0)]["R_raw"] == pytest.approx(0.4)


def test_arm_scores_and_unknown_fallback():
    d = np.asarray([0.2, 0.5, 0.9, 0.1])
    keys = [KX, KY, KX, "9|9|9|9|9"]
    degree = np.asarray([2, 2, 3, 2])
    erc = np.asarray([0, 0, 1, 0])
    tables = {
        "sig": {KX: {"R_shrunk": 0.2}, KY: {"R_shrunk": -0.1}},
        "ctx": {bcv.context_key(KX, 3, 1): {"R_shrunk": 0.05}},
    }
    sig_b, ctx_b = bcv.bonus_arrays(tables, keys, degree, erc)
    assert np.allclose(sig_b, [0.2, -0.1, 0.2, 0.0])
    assert np.allclose(ctx_b, [0.0, 0.0, 0.05, 0.0])  # unknown signature gets no context bonus
    assert np.allclose(bcv.arm_score(d, sig_b, ctx_b, "A"), d)
    assert np.allclose(bcv.arm_score(d, sig_b, ctx_b, "B"), d + sig_b)
    assert np.allclose(bcv.arm_score(d, sig_b, ctx_b, "C"), d + sig_b + ctx_b)


def test_topk_ties_and_metrics():
    delta = np.asarray([0.1, 0.9, 0.4, 0.6])
    score = np.asarray([1.0, 1.0, 0.0, 0.0])
    # stable tie-break: indices 0, 1
    assert bcv.top_k_indices(score, 2).tolist() == [0, 1]
    assert bcv.topk_mean_abs(delta, score, 2) == pytest.approx((0.1 + 0.9) / 2.0)
    oracle = np.asarray([0.0, 1.0, 0.0, 1.0])
    assert bcv.topk_oracle_overlap(score, oracle, 2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# conditional pseudo-control
# ---------------------------------------------------------------------------


def test_pseudo_preserves_dev_strata_and_is_deterministic():
    dev = np.linspace(0.0, 1.0, 8)
    ig = np.asarray([0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 1.0, 1.1])
    residual = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    js = _js(0, 7, [KX] * 8, [2] * 8, [0] * 8, dev, ig, residual=residual)
    p1 = bcv.pseudo_junction_sets([js], 0)
    p2 = bcv.pseudo_junction_sets([js], 0)
    assert np.allclose(p1[0].r, p2[0].r)
    bins = np.minimum((js.d * bcv.PSEUDO_DEV_BINS).astype(int), bcv.PSEUDO_DEV_BINS - 1)
    for b in range(bcv.PSEUDO_DEV_BINS):
        idx = np.where(bins == b)[0]
        assert sorted(p1[0].r[idx].tolist()) == sorted(residual[idx].tolist())
    assert np.allclose(p1[0].d, js.d)
    assert list(p1[0].keys) == list(js.keys)
    # different replicates generally differ
    reps = [bcv.pseudo_junction_sets([js], p)[0].r for p in range(8)]
    assert any(not np.allclose(reps[0], r) for r in reps[1:])


def test_pseudo_keeps_support_counts():
    js = []
    for row in range(6):
        keys = [KX] * 3 + [KY] * 3
        dev = [0.1, 0.4, 0.9, 0.2, 0.6, 0.8]
        ig = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        js.append(_js(0, row, keys, [2] * 6, [0] * 6, dev, ig))
    base = bcv.build_residual_tables(js, signature_floor=2, context_floor=2)
    for p in range(4):
        tbl = bcv.build_residual_tables(bcv.pseudo_junction_sets(js, p), signature_floor=2, context_floor=2)
        assert {k: v["n_molecules"] for k, v in tbl["sig"].items()} == {
            k: v["n_molecules"] for k, v in base["sig"].items()
        }


def test_compact_tables_roundtrip():
    js = [_js(0, r, [KX, KY], [2, 3], [0, 1], [0.2, 0.8], [0.4, 0.9]) for r in range(12)]
    tables = bcv.pseudo_tables(js, replicates=3, signature_floor=2, context_floor=2)
    compact = bcv.compact_tables(tables)
    for i in range(3):
        view = bcv.tables_from_compact(compact, i)
        for k in compact["sig_keys"]:
            assert view["sig"][k]["R_shrunk"] == tables[i]["sig"][k]["R_shrunk"]
        for k in compact["ctx_keys"]:
            assert view["ctx"][k]["R_shrunk"] == tables[i]["ctx"][k]["R_shrunk"]


# ---------------------------------------------------------------------------
# statistics are molecular units
# ---------------------------------------------------------------------------


def test_contrast_uses_molecule_units():
    a = np.asarray([1.0, 0.0, 0.5])
    b = np.asarray([0.5, 0.0, 0.5])
    out = bcv.contrast(a, b)
    assert out["n_molecules"] == 3
    assert out["mean"] == pytest.approx(0.5 / 3.0)
    # duplicating one molecule's row must not act as extra independent evidence:
    # the statistic is a mean over the rows it is given, so callers must pass
    # one row per molecule (the merged_seed_arrays helper enforces that contract)
    merged = bcv.merged_seed_arrays([np.asarray([1.0, 0.0]), np.asarray([0.0, 0.0])])
    assert np.allclose(merged, [0.5, 0.0])
    assert bcv.contrast(merged, np.zeros(2))["n_molecules"] == 2


def test_merged_seed_arrays_is_mean_not_concatenation():
    s0 = np.asarray([1.0, 2.0])
    s1 = np.asarray([3.0, 4.0])
    assert np.allclose(bcv.merged_seed_arrays([s0, s1]), [2.0, 3.0])
    assert bcv.merged_seed_arrays([s0, s1]).shape == (2,)


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------


def _gates(**over):
    g = {f"A{i}": True for i in range(1, 9)}
    g.update({f"B{i}": True for i in range(1, 6)})
    g.update(over)
    return g


def test_decision_rule_four_labels():
    assert bcv.decide_label(_gates()) == "SIGNATURE_HAS_CONDITIONAL_VALUE"
    # context-only: B-A statistically unsupported, C-B fully supported
    g = _gates(A1=False, A2=False, A3=False)
    assert g["B1"] and g["B5"]
    assert bcv.decide_label(g) == "CONTEXT_ADDS_VALUE_BUT_SIGNATURE_WEAK"
    # positive but CI crosses zero and context does not rescue -> inconclusive
    assert bcv.decide_label(_gates(A1=True, A2=False, B1=True, B2=False)) == "INCONCLUSIVE"
    # statistically reliable but practically marginal -> inconclusive
    assert bcv.decide_label(_gates(A8=False, B5=False)) == "INCONCLUSIVE"
    # statistically reliable but fails pseudo-control -> no confirmed value
    assert bcv.decide_label(_gates(A6=False)) == "NO_CONFIRMED_CONDITIONAL_VALUE"
    # statistically reliable but concentrated in one class -> no confirmed value
    assert bcv.decide_label(_gates(A5=False)) == "NO_CONFIRMED_CONDITIONAL_VALUE"
    # non-positive
    assert bcv.decide_label(_gates(A1=False, B1=False)) == "NO_CONFIRMED_CONDITIONAL_VALUE"


def test_support_histogram_buckets():
    hist = bcv.support_histogram([3, 12, 25, 70, 150, 400])
    assert hist == {"<10": 1, "10-19": 1, "20-49": 1, "50-99": 1, "100-199": 1, ">=200": 1}


# ---------------------------------------------------------------------------
# discipline / freeze integrity
# ---------------------------------------------------------------------------


def test_no_label_or_test_loader_tokens():
    module = (REPO_ROOT / "tracks/structure-contribution/code/bjoc_signature_conditional_value_v1.py").read_text(encoding="utf-8")
    runner = (REPO_ROOT / "tracks/structure-contribution/code/run_bjoc_signature_conditional_value_v1.py").read_text(encoding="utf-8")
    for text in (module, runner):
        assert "objects_test" not in text
        assert "locked_y" not in text
        assert "valid_y" not in text
        assert "monitor_read" not in text
        assert "train_supervised" not in text
        assert "torch.optim" not in text
    assert '"labels_read": False' in module or "'labels_read': False" in module
    assert '"model_inference": False' in module or "'model_inference': False" in module
    assert "bcv.DISCIPLINE" in runner
    assert "official_test_loaded" in module
    # the frozen evidence base itself
    assert "phase0_results" in runner and "sample_manifest" in runner
    assert "objects_valid" in runner and "objects_test" not in runner


def test_runner_parser_commands_and_constants():
    import run_bjoc_signature_conditional_value_v1 as runner

    parser = runner.build_parser()
    commands = {a.dest: a.choices for a in parser._actions if a.dest == "cmd"}["cmd"]
    for name in ("verify-inputs", "freeze", "prepare", "residuals", "pseudo", "crossfit", "analyze", "finalize"):
        assert name in commands
    assert bcv.TOP_K_PRIMARY == 4
    assert bcv.PSEUDO_REPLICATES == 16
    assert bcv.BOOTSTRAP_N == 5000
    assert bcv.RECOVERY_FRACTION_MIN == 0.05
    assert runner.PRIMARY_K == 4
    assert set(runner.KS) == {1, 2, 4, 8}


def test_fit_input_is_discovery_only_by_construction():
    """build_residual_tables consumes exactly the junction sets it is given."""
    disc = [
        _js(0, 1, [KX], [2], [0], [0.0], [0.0], residual=[0.5]),
        _js(1, 1, [KX], [2], [0], [0.0], [0.0], residual=[0.5]),
    ]
    extra = [_js(0, 999, [KX], [2], [0], [0.0], [0.0], residual=[-0.5])]
    t_disc = bcv.build_residual_tables(disc, signature_floor=2, context_floor=2)
    t_extra = bcv.build_residual_tables(disc + extra, signature_floor=2, context_floor=2)
    assert t_disc["sig"][KX]["R"] == pytest.approx(0.5)
    assert t_extra["sig"][KX]["R"] == pytest.approx((0.5 - 0.5) / 2.0)
    assert t_disc["sig"][KX]["n_molecules"] == 1
    assert t_extra["sig"][KX]["n_molecules"] == 2
