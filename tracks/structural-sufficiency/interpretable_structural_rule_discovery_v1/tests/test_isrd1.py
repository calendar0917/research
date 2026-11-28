"""Targeted tests for the ISRD-v1 repairs and deliverables.

Each test targets a known v0 failure mode or a required v1 identity check:

1. arm-specific structural column mapping, including the X-F ``f2258`` boundary;
2. mapped structure values equal the booster's own structural input columns;
3. several thresholds on one structure are not collapsed;
4. every real leaf path's thresholds and branch directions reproduce the tree;
5. a single-granularity booster never yields a C–F real path;
6. real and matched random joint rules pass the *exact* main-effect eligibility;
7. raw integer counts and scaled-count semantics;
8. B2 / B0 historical prediction replay;
9. R128 and COUNT128 use identical structure identities;
10. COUNT-ALL and XGB-ALL share the same input identity;
11. raw-unit closure of the interpretable linear predictions;
12. evaluation metrics recomputed independently from stored predictions;
13. sparse centred logistic == dense standardised logistic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "code"))

import isrd1_common as K  # noqa: E402
import isrd1_rules as R  # noqa: E402


@pytest.fixture(scope="module")
def data() -> K.Data:
    return K.Data()


@pytest.fixture(scope="module")
def phaseA() -> dict:
    return K.read_json(ROUND / "analysis" / "phaseA_rules.json")


@pytest.fixture(scope="module")
def phaseBC() -> dict:
    return K.read_json(ROUND / "analysis" / "phaseBC.json")


@pytest.fixture(scope="module")
def phaseC() -> dict:
    return K.read_json(ROUND / "analysis" / "phaseC.json")


# ---------------------------------------------------------------------------
# 1. arm-specific mapping
# ---------------------------------------------------------------------------


def test_decode_feature_arm_specific_boundaries():
    assert K.H_WIDTH == 2258
    assert K.ARM_WIDTH == {"X-C": 3012, "X-F": 5910}
    assert K.decode_feature("X-C", 2257) == ("H", 2257)
    assert K.decode_feature("X-C", 2258) == ("C", 0)
    assert K.decode_feature("X-F", 2258) == ("F", 0)          # the v0 error
    assert K.decode_feature("X-C", 3011) == ("C", 753)
    assert K.decode_feature("X-F", 5909) == ("F", 3651)
    for bad in (("X-C", 3012), ("X-F", 5910), ("X-F", -1), ("X-C", -1)):
        with pytest.raises(ValueError):
            K.decode_feature(*bad)
    # no X-F structural index may ever decode to COARSE
    for i in (0, 1, 753, 754, 3651):
        assert K.decode_feature("X-F", K.H_WIDTH + i)[0] == "F"


def test_booster_widths_and_decoded_granularity():
    for arm, (gran, _bw) in K.ARM_BLOCK.items():
        b = R.load_booster(arm)
        assert R.booster_width(b) == K.ARM_WIDTH[arm]
        nodes = R.split_nodes(b, arm)
        assert nodes, "no structural split nodes"
        assert {nd["granularity"] for nd in nodes} == {gran}
        # the layout asserted by the protocol is not the v0 H+C+F layout
        for nd in nodes[:50]:
            assert nd["feature"] - K.H_WIDTH < K.ARM_BLOCK[arm][1]


# ---------------------------------------------------------------------------
# 2. mapped values == booster structural columns
# ---------------------------------------------------------------------------


def test_mapping_report_and_direct_column_values(data):
    mapping = K.read_json(ROUND / "analysis" / "phaseA_mapping.json")
    assert mapping["h_width"] == 2258
    assert not mapping["h_has_nan"]
    for arm, rec in mapping["arms"].items():
        assert rec["booster_num_features"] == K.ARM_WIDTH[arm]
        assert rec["max_abs_value_diff_mapped_vs_booster_column"] == 0.0
        assert rec["n_features_value_checked"] > 0
    assert mapping["arms"]["X-F"]["boundary_f2258"]["decoded"] == ["F", 0]
    assert mapping["arms"]["X-F"]["boundary_f2258"]["matches_block_col0"] == 0.0
    # the F[0] column is different from C[0] in general, so the boundary test bites
    assert data.sub32("F", 0, data.rows["train"][:500]).max() != pytest.approx(
        float(data.sub32("C", 0, data.rows["train"][:500]).max()), abs=1e-9)


def test_leaf_replay_matches_booster(data):
    mapping = K.read_json(ROUND / "analysis" / "phaseA_mapping.json")
    for arm, rec in mapping["path_replay"].items():
        assert rec["n_leaf_mismatch"] == 0
        assert rec["trees_checked"] >= 10
        assert not rec["missing_branch_used"]


# ---------------------------------------------------------------------------
# 3. multiple thresholds per structure survive
# ---------------------------------------------------------------------------


def test_multiple_thresholds_per_structure_preserved(phaseA):
    singles = phaseA["singles"]
    by_struct: dict[tuple[str, int], set] = {}
    for c in singles:
        by_struct.setdefault((c["granularity"], int(c["column"])), set()).add(
            (R.f32(c["threshold"]), c["direction"]))
    multi = {k: v for k, v in by_struct.items() if len({t for t, _d in v}) > 1}
    assert multi, "no structure carries more than one threshold (suspicious)"
    # a concrete case: the same structure, two thresholds, both kept
    key, vals = max(multi.items(), key=lambda kv: len(kv[1]))
    thresholds = sorted({t for t, _d in vals})
    assert len(thresholds) >= 2
    with_thr = [c for c in singles if (c["granularity"], int(c["column"])) == key]
    assert {R.f32(c["threshold"]) for c in with_thr} == set(thresholds)
    # both directions of the same node are separate conditions, not overwritten
    assert {c["direction"] for c in with_thr} >= {"ge"}


def test_conditions_carry_source_nodes(phaseA):
    for c in phaseA["singles"][:50]:
        assert c["sources"]
        for arm, rec in c["sources"].items():
            assert rec["n_nodes"] >= 1 and rec["n_trees"] >= 1
        assert c["n_nodes"] >= 1


# ---------------------------------------------------------------------------
# 4. real path conditions reproduce the tree
# ---------------------------------------------------------------------------


def test_real_path_conditions_reproduce_leaf_assignment(data):
    import xgboost as xgb

    sys.path.insert(0, str(K.SGA / "code"))
    from run_phaseA_xgb import design_rows, unpack_h

    with np.load(K.BUNDLE) as z:
        bundle = {k: np.asarray(z[k]) for k in z.files}
    h = unpack_h(bundle)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(data.rows["train"], size=400, replace=False))
    for arm in ("X-C", "X-F"):
        gran, _ = K.ARM_BLOCK[arm]
        b = R.load_booster(arm)
        X = design_rows(h, [(gran, data.block[gran])], rows)
        leaf = b.predict(xgb.DMatrix(X), pred_leaf=True).astype(np.int64)
        paths = R.path_conditions(b, arm)
        for tree in range(6):
            leaves = {k: v for k, v in paths.items() if k[0] == tree}
            n_checked = 0
            for (_t, leaf_id), rec in leaves.items():
                # rows that really reach this leaf according to xgboost
                take = leaf[:, tree] == leaf_id
                if not take.any():
                    continue
                for cond in rec["conditions"]:
                    v = X[take, cond["feature"]]
                    thr = np.float32(cond["threshold"])
                    ok = (v < thr) if cond["direction"] == "lt" else (v >= thr)
                    assert ok.all(), (arm, tree, leaf_id, cond)
                    n_checked += 1
                # every condition names its real source node on the path
                for cond in rec["conditions"]:
                    assert cond["tree"] == tree and cond["arm"] == arm
                    assert cond["direction"] in ("lt", "ge")
                    assert cond["missing_goes"] in ("yes", "no")
                    assert cond["granularity"] == K.ARM_BLOCK[arm][0]
            assert n_checked > 0
            # structural conditions alone need not separate leaves: H splits are
            # deliberately dropped from the projection.  Leaves sharing the same
            # structural condition set must differ by at least one H split.
            sets = {}
            for (_t, leaf_id), rec in leaves.items():
                key = tuple((c["feature"], c["direction"], c["threshold"])
                            for c in rec["conditions"])
                sets.setdefault(key, []).append(leaf_id)
            for key, ids in sets.items():
                if len(ids) > 1:
                    assert any(rec["n_h_splits"] > 0 for (_t, lid), rec in leaves.items()
                               if lid in ids), "duplicate structural paths without H splits"


# ---------------------------------------------------------------------------
# 5. no mixed C–F real path
# ---------------------------------------------------------------------------


def test_no_mixed_granularity_real_pairs(phaseA):
    assert phaseA["pair_stats"]["n_mixed_granularity_pairs"] == 0
    for p in phaseA["pairs"]:
        assert p["gran_pair"] in ("C+C", "F+F")
        assert p["key_a"][0] == p["key_b"][0] == p["gran_pair"][0]
    # v0 produced 38 C+F pairs from the same two boosters; impossible here
    v0 = K.read_json(K.V0 / "analysis" / "phaseA_rules.json")
    assert any(p["gran_pair"] == "C+F" for p in v0["pairs"])


# ---------------------------------------------------------------------------
# 6. exact main-effect eligibility for real and random pairs
# ---------------------------------------------------------------------------


def test_joint_main_effect_eligibility_is_exact(phaseBC):
    joint = phaseBC["phaseC"]["joint"]
    assert joint["status"] == "EXECUTED"
    u64_keys = set()
    for rid, thr, direction in zip(phaseBC["phaseC"]["r64_rule_ids"],
                                   phaseBC["phaseC"]["r64_thresholds"],
                                   phaseBC["phaseC"]["r64_directions"]):
        rec = next(s for s in K.read_json(ROUND / "analysis" / "phaseA_rules.json")["singles"]
                   if s["rule_id"] == rid)
        u64_keys.add((rec["granularity"], int(rec["column"]), R.f32(thr), direction))
    closure = set(u64_keys) | {R.complement(k) for k in u64_keys}

    def parse(condition: str) -> tuple:
        head, op, val = condition.split()
        gran, col = head[0], int(head[2:-1])
        return (gran, col, R.f32(float(val)), "ge" if op == ">=" else "lt")

    sel = joint["selected"]
    assert sel["n_pairs"] >= 8
    for rec in sel["real_pairs"] + sel["random_pairs"]:
        assert parse(rec["condition_a"]) in closure
        assert parse(rec["condition_b"]) in closure
        assert len({(parse(rec["condition_a"])[0], parse(rec["condition_a"])[1]),
                    (parse(rec["condition_b"])[0], parse(rec["condition_b"])[1])}) == 2
    # the exact identity check bites: a nearby different threshold is not eligible
    k = next(iter(u64_keys))
    assert (k[0], k[1], R.f32(k[2] * 1.01), k[3]) not in closure
    assert R.complement(k) in closure
    m = joint["match"]["stats"]
    assert m["tolerances"] == {"marginal_abs_freq": 0.05, "joint_abs_freq": 0.03}
    assert m["max_abs_diff"]["freq_a"] <= 0.05 + 1e-12
    assert m["max_abs_diff"]["freq_b"] <= 0.05 + 1e-12
    assert m["max_abs_diff"]["freq_joint"] <= 0.03 + 1e-12
    assert m["one_to_one_without_replacement"] is True


# ---------------------------------------------------------------------------
# 7. raw count semantics
# ---------------------------------------------------------------------------


def test_raw_count_and_scaled_count_semantics(data):
    rng = np.random.default_rng(0)
    for gran, n in (("C", K.N_C), ("F", K.N_F)):
        for col in rng.choice(n, size=6, replace=False):
            x = data.col(gran, int(col)).astype(np.float64)
            raw = data.raw_count(gran, int(col))
            back = np.rint(x * np.sqrt(data.n_atoms)).astype(np.int64)
            assert np.array_equal(back, raw), (gran, col)
            assert np.all(raw >= 0)
            assert raw.max() > 0


def test_threshold_reading_is_empirical(phaseA, data):
    singles = phaseA["singles"]
    checked = 0
    for c in singles:
        rd = c["reading"]
        if rd.get("train_equals_raw_ge_kmin"):
            raw = data.raw_count(c["granularity"], int(c["column"]))[data.rows["train"]]
            k = int(rd["k_min_active"])
            act = R.rule_activation(data, c, data.rows["train"]).astype(bool)
            assert np.array_equal(act, raw >= k)
            checked += 1
    assert checked >= 1, "no empirically readable threshold found"
    # the presence reading must not be claimed as a global logical equivalence
    assert all("empirically" not in str(c.get("reading", {}).get("claim", ""))
               for c in singles[:5])


# ---------------------------------------------------------------------------
# 8. historical replay
# ---------------------------------------------------------------------------


def test_b2_replay(phaseBC):
    rec = phaseBC["b2_replay"]
    assert rec["max_abs_diff"] < 1e-6
    assert rec["n_coefficients"] == 6456


def test_b0_replay(data):
    coef = np.load(K.B2_COEF)["B0|0.03"].astype(np.float64)
    assert coef.size == 2050
    rec = K.read_json(K.MSDS / "analysis" / "phaseB_linear.json")
    intercept = None
    for f in rec["arms"]["B0"]["fits"]:
        if abs(float(f["candidate_C"]) - 0.03) < 1e-12:
            intercept = float(f["coefficients"]["intercept"])
    assert intercept is not None
    with np.load(K.MORGAN) as zm:
        bits = np.unpackbits(zm["packed"], axis=1)[:, : int(zm["n_bits"])].astype(np.float64)
    a = np.log1p(data.n_atoms.astype(np.float64))
    b = np.log1p(data.n_bonds.astype(np.float64))
    tr = data.rows["train"]
    za = (a - a[tr].mean()) / a[tr].std()
    zb = (b - b[tr].mean()) / b[tr].std()
    X = np.column_stack([bits, za, zb])
    logit = intercept + X @ coef
    del X
    with np.load(K.B2_PRED) as z:
        saved = {k: z[f"B0|0.03|{k}"].astype(np.float64) for k in K.SPLITS}
    worst = max(float(np.max(np.abs(logit[data.rows[k]] - saved[k]))) for k in K.SPLITS)
    assert worst < 1e-6, worst


# ---------------------------------------------------------------------------
# 9/10. identity of inputs across arms
# ---------------------------------------------------------------------------


def test_r128_and_count128_share_structures(phaseA, phaseBC, phaseC):
    singles = {c["rule_id"]: c for c in phaseA["singles"]}
    ids = phaseBC["phaseB"]["selected_rule_ids"][:128]
    expected = [[singles[i]["granularity"], int(singles[i]["column"])] for i in ids]
    assert phaseC["count128_structures"] == expected
    assert phaseC["count128_rule_ids"] == ids
    assert len({tuple(s) for s in expected}) == len(expected)


def test_count_all_and_xgb_all_input_identity(data, phaseC):
    from run_phaseC import dense_design

    rows = data.rows["valid"][:257]
    dense = dense_design(data, rows)
    assert dense.shape[1] == phaseC["input_identity"]["count_all_width"]
    assert dense.dtype == np.float32
    stat = data.all_structure_matrix()[rows].toarray()
    assert np.array_equal(dense[:, 2:], stat.astype(np.float32))
    assert np.array_equal(dense[:, 0], np.log1p(data.n_atoms.astype(np.float64))[rows].astype(np.float32))
    assert np.array_equal(dense[:, 1], np.log1p(data.n_bonds.astype(np.float64))[rows].astype(np.float32))
    assert phaseC["xgb_all"]["design_width"] == dense.shape[1]


# ---------------------------------------------------------------------------
# 11. raw-unit closure
# ---------------------------------------------------------------------------


def test_count_all_raw_unit_closure(data, phaseC):
    arm = phaseC["count_all"]
    w = np.asarray(arm["coef_raw"], dtype=np.float64)
    b = float(arm["intercept_raw"])
    X = sp.hstack([sp.csr_matrix(data.size_matrix()), data.all_structure_matrix()], format="csr")
    z = b + np.asarray(X @ w).ravel()
    with np.load(ROUND / "analysis" / "predictions" / "COUNT-ALL.npz") as zz:
        logit = zz["logit"]
    assert float(np.max(np.abs(z - logit))) < 1e-8
    assert arm["closure_raw_units"] < 1e-8


def test_count128_raw_unit_closure(data, phaseC):
    arm = phaseC["count128"]
    w = np.asarray(arm["coef_raw"], dtype=np.float64)
    X = np.column_stack([data.size_matrix()] + [
        data.col(g, int(c)).astype(np.float64) for g, c in phaseC["count128_structures"]])
    z = float(arm["intercept_raw"]) + X @ w
    with np.load(ROUND / "analysis" / "predictions" / "COUNT128.npz") as zz:
        assert float(np.max(np.abs(z - zz["logit"]))) < 1e-8


def test_r128_rule_closure(data, phaseBC, phaseA):
    singles = {c["rule_id"]: c for c in phaseA["singles"]}
    arm = phaseBC["phaseB"]["arms"]["R128"]
    coef = np.asarray(arm["coef"], dtype=np.float64)
    ids = phaseBC["phaseB"]["selected_rule_ids"][:128]
    rules = [singles[i] for i in ids]
    acts = R.rule_matrix(data, rules)
    size = data.size_matrix()
    smu, ssd = np.asarray(phaseBC["size_standardisation"]["mean"]), np.asarray(
        phaseBC["size_standardisation"]["std"])
    rmu = np.asarray(phaseBC["phaseB"]["rule_standardisation"]["mean"])[:128]
    rsd = np.asarray(phaseBC["phaseB"]["rule_standardisation"]["std"])[:128]
    z = (float(arm["intercept"]) - np.sum(coef[:2] * smu / ssd) - np.sum(coef[2:] * rmu / rsd)
         + size @ (coef[:2] / ssd) + acts @ (coef[2:] / rsd))
    with np.load(ROUND / "analysis" / "predictions" / "R128.npz") as zz:
        assert float(np.max(np.abs(z - zz["logit"]))) < 1e-8


# ---------------------------------------------------------------------------
# 12. metrics independently recomputed
# ---------------------------------------------------------------------------


def test_metrics_recomputed(phaseC, data):
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = pd.read_csv(ROUND / "analysis" / "metrics.csv")
    assert not df.empty
    worst = 0.0
    n = 0
    for _, row in df.iterrows():
        path = ROUND / "analysis" / "predictions" / f"{str(row['arm']).replace('+', '_')}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            logit = z["logit"]
        r = data.rows[row["split"]]
        keep = np.isfinite(logit[r])
        if keep.sum() == 0:
            continue
        r = r[keep]
        p = K.sigmoid(logit[r])
        worst = max(worst, abs(roc_auc_score(data.y[r], p) - float(row["roc_auc"])),
                    abs(average_precision_score(data.y[r], p) - float(row["average_precision"])))
        n += 1
    assert n >= 10
    assert worst < 1e-9


def test_development_test_label(phaseC):
    import pandas as pd

    df = pd.read_csv(ROUND / "analysis" / "metrics.csv")
    test_rows = df[df["split"] == "test"]
    assert len(test_rows) > 0
    assert set(test_rows["evidence_label"].unique()) == {"DEVELOPMENT_TEST"}


# ---------------------------------------------------------------------------
# 13. sparse centred logistic == dense standardised logistic
# ---------------------------------------------------------------------------


def test_sparse_logistic_matches_dense():
    rng = np.random.default_rng(0)
    n, p = 4000, 25
    X = (rng.random((n, p)) < 0.25) * rng.integers(1, 4, size=(n, p))
    Xs = sp.csr_matrix(X.astype(np.float64))
    y = (rng.random(n) < 0.2).astype(np.float64)
    tr = np.arange(3000)
    sw = np.zeros(n)
    sw[tr] = K.class_balanced_weights(y[tr])
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    for C in (0.03, 0.3):
        dense = K.fit_logistic(Z[tr], y[tr], sw[tr], C)
        sparse = K.fit_logistic_sparse_standardised(Xs, y, sw, C, mu, sd)
        assert np.max(np.abs(dense["coef"] - sparse["coef"])) < 1e-6
        assert abs(dense["intercept"] - sparse["intercept"]) < 1e-6
        assert np.max(np.abs(dense["logit"] - sparse["logit"][tr])) < 1e-8
