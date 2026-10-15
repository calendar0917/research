"""Fast correctness tests for ``bjoc_source_attribution_feasibility_v1``.

No ZINC data and no frozen checkpoint are required for the core tests: tiny
synthetic molecules and a randomly initialised BJOC only.  Source/manifest
tests skip cleanly when the git-ignored binaries or generated artifacts are
absent.

Covers: the exact one-at-a-time IG identity ``IG_j = -occ_j`` (two baselines),
junction-permutation equivariance, identical-row symmetry, the ERRATUM
statistic distinction (``mean(abs)`` vs ``abs(mean)``), the exact sign-test,
the Organized-proper vs Organized-v1-style baseline semantics, sample-rule
reuse, and (when generated) attribution/metrics consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_source_attribution_feasibility_v1 as sa  # noqa: E402
import bjoc_structure_explanation_audit_v1 as aud  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _mol(idx: int, atom_types, bonds):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(idx, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))


def _tiny_world():
    specs = [
        ([0, 1, 2, 3], [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 1)]),
        ([4, 5, 6], [(0, 1, 1), (1, 2, 3)]),
        ([7, 7, 8, 9, 10], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 1, 1)]),
        ([1, 2, 3, 4, 5], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 0, 1)]),
        ([2, 2, 3], [(0, 1, 2), (1, 2, 1)]),
        ([0, 1, 2, 3, 4, 5], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 1), (5, 0, 1), (0, 3, 1)]),
    ]
    return [_mol(i, a, b) for i, (a, b) in enumerate(specs)]


def _model(seed: int = 0) -> src.BJOC:
    model = src.build_bjoc(
        src.BJOCConfig(), seed=seed, organization_mode=src.NULL,
        global_mean=NORM["global_mean"], global_scale=NORM["global_scale"],
    )
    model.eval()
    return model


def _packs():
    objs = _tiny_world()
    batch = sro.batch_objects(objs)
    model = _model()
    feats = aud.run_features(model, batch)
    for g in range(len(objs)):
        pack = sa.pack_from_features(model, batch, feats, g)
        if pack["n_junctions"] >= 2:
            yield model, pack


def test_ig_identity_graph_mean_baseline():
    checked = 0
    for model, pack in _packs():
        mu = pack["w"].mean(0)
        ig = sa.ig_scores(model, pack, mu)
        occ = sa.occlusion_scores(model, pack, mu)
        scale = float(occ.abs().mean())
        if scale < 1e-9:
            continue
        err = float((ig["ig"] + occ).abs().max()) / scale
        assert err <= 0.05, err
        assert aud.spearman(ig["ig"].numpy(), -occ.numpy()) >= 0.8
        checked += 1
    assert checked >= 2


def test_ig_identity_zero_baseline():
    checked = 0
    for model, pack in _packs():
        zero = torch.zeros_like(pack["w"][0])
        ig = sa.ig_scores(model, pack, zero)
        occ = sa.occlusion_scores(model, pack, zero)
        scale = float(occ.abs().mean())
        if scale < 1e-9:
            continue
        assert float((ig["ig"] + occ).abs().max()) / scale <= 0.05
        checked += 1
    assert checked >= 2


def test_ig_permutation_equivariance():
    objs = _tiny_world()
    batch = sro.batch_objects(objs)
    model = _model()
    feats = aud.run_features(model, batch)
    for g in range(len(objs)):
        pack = sa.pack_from_features(model, batch, feats, g)
        if pack["n_junctions"] < 2:
            continue
        ig = sa.ig_scores(model, pack, pack["w"].mean(0))["ig"]
        perm_batch, order = sa.permute_junctions_with_perm(batch, seed=11 + g)
        perm_feats = aud.run_features(model, perm_batch)
        perm_pack = sa.pack_from_features(model, perm_batch, perm_feats, g)
        ig_perm = sa.ig_scores(model, perm_pack, perm_pack["w"].mean(0))["ig"]
        j_off = int(torch.nonzero(batch["junc_batch"] == g)[0, 0])
        local_order = order[j_off : j_off + pack["n_junctions"]] - j_off
        assert float((ig_perm.numpy() - ig.numpy()[local_order]).__abs__().max()) < 1e-5


def test_ig_symmetry_for_identical_rows():
    checked = 0
    for model, pack in _packs():
        w = pack["w"].clone()
        w[1] = w[0]
        scores = sa.ig_scores(model, {**pack, "w": w}, w.mean(0))["ig"]
        assert abs(float(scores[0]) - float(scores[1])) <= 1e-6
        checked += 1
    assert checked >= 2


def test_errata_statistic_distinction_and_sign_test():
    d = np.asarray([0.5, 0.5, -0.4], dtype=np.float64)
    assert sa.mean_abs(d) > abs(float(d.mean()))  # mean(abs) != abs(mean)
    assert sa.exact_binomial_p(10, 10) == pytest.approx(2.0 / 1024.0, rel=1e-12)
    assert sa.exact_binomial_p(5, 10) == pytest.approx(1.0, rel=1e-12)
    assert sa.exact_binomial_p(9, 10) == pytest.approx(2 * 11 / 1024.0, rel=1e-9)


def test_organized_proper_baseline_is_not_null_baseline():
    diffs = []
    for model, pack in _packs():
        w = pack["w"]
        delta = w.std(dim=0, unbiased=False)
        w2 = w.clone()
        w2[0] = w2[0] + delta
        with torch.no_grad():
            null_nat = float(sa.predict_null(model, pack))
            org_nat = float(sa.predict_organized(model, pack, w))
            org_pert = float(sa.predict_organized(model, pack, w2))
        org_proper = org_pert - org_nat
        org_v1_style = org_pert - null_nat
        diffs.append(abs(org_proper - org_v1_style))
        assert abs(org_nat - null_nat) > 1e-9  # the two modes differ at the natural point
    assert max(diffs) > 1e-9
    assert len(diffs) >= 2


def test_sample_manifest_reuses_v1_rule():
    new_manifest = sa.OUT / "sample_manifest.json"
    v1_manifest = sa.V1_DIR / "sample_manifest.json"
    if not new_manifest.exists() or not v1_manifest.exists():
        pytest.skip("generated sample manifests absent")
    new = aud.read_json(new_manifest)
    old = aud.read_json(v1_manifest)
    assert new["diagnostic_rows"] == old["diagnostic_rows"]
    assert new["smoke_rows"] == old["smoke_rows"]
    assert new["reference_rows"] == old["reference_rows"]
    assert new["diagnostic_key_sha256"] == old["diagnostic_key_sha256"]
    assert new["official_valid_loaded"] is False and new["official_test_loaded"] is False


def test_source_manifest_hash_and_reproduction_gate():
    path = sa.OUT / "source_manifest.json"
    if not path.exists():
        pytest.skip("generated source manifest absent")
    manifest = aud.read_json(path)
    assert manifest["historical_hash_checks"]["soups_and_cache_match_records"] is True
    assert manifest["reproduction_pass"] is True
    for seed in ("0", "1"):
        gate = manifest["reproduction_gate"][seed]
        assert gate["params"] == 413505
        assert gate["reader_in_dim"] == 321
        assert gate["forward_vs_reader_max_abs"] == 0.0


def test_attribution_metrics_consistency():
    attr_path = sa.OUT / "attribution.json"
    metrics_path = sa.OUT / "metrics.json"
    if not attr_path.exists() or not metrics_path.exists():
        pytest.skip("generated attribution artifacts absent")
    attr = aud.read_json(attr_path)
    metrics = aud.read_json(metrics_path)
    for seed in ("0", "1"):
        r = attr["per_seed"][seed]
        assert r["pooled_spearman_ig_mean_vs_neg_occ_mean"] >= 0.99
        for alpha in ("alpha_1.0", "alpha_0.5"):
            t = r["top_vs_random"][alpha]
            expected = t["mean"] > 0 and t["median"] > 0 and t["p_two_sided_exact"] < 0.01
            assert expected is True
            assert metrics["top_vs_random"][seed][alpha]["p_two_sided_exact"] == t["p_two_sided_exact"]
    assert metrics["decisions"]["METHOD_VALIDATED"] is True
    assert metrics["decisions"]["TOP_JUNCTIONS_RESPONSE_EXCEEDS_RANDOM"] is True
    assert metrics["decisions"]["TOP_JUNCTIONS_FORM_CONNECTED_COMBINATIONS"] is False
    assert metrics["source_manifest"]["reproduction_pass"] is True
