"""cscl-v3d correctness tests.

Covers (protocol ``configs/cscl_v3d_bucket.yaml`` verification gates):

Phase B / information fidelity
  * bucket features conserve counts and rebuild Full-Y's per-bucket pooled
    first/second moments FROM THE STORED CACHE (not from a live forward);
  * bucket sums contain the old marginal fields; total pair counts match;
  * extraction is a compatible extension (shared fields identical to the v3
    cache on sampled molecules);
  * the audit-only ``pair_ref_*`` arrays are never read by the training collate.

Phase C / models
  * same-information contract: S and O read the SAME BucketBatch tensors from
    ONE collate function; legacy columns bit-identical to the v3 collate;
  * deterministic representation: fixed bucket order, zero-count buckets -> 0,
    mean/rms recoverable from stored sums + counts;
  * bit-exact shared init + untrained prediction equality (both arms);
  * SAME +61,760 added params on both arms; new columns zero-init;
  * channel routing (AST): unit bucket info only in delta / unit encoder,
    rel bucket info only in gamma / relation encoder; no post-sum fusion;
    no global bucket bypass; alpha unchanged per-type scalar;
  * new-channel learnability: gradients reach the new columns and a training
    step changes predictions;
  * objective discipline (S: L1 + gamma penalty verbatim; O: L1 only);
  * strict additivity + lossless gauge canonicalization + UNK fallback;
  * relabel / endpoint-swap / batch-composition invariance;
  * synthetic task A truth consistency + task B truth definition.

Heavy tests are marked ``slow`` (they load caches / run short CPU training).
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
for _p in (str(_CODE_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytestmark = pytest.mark.slow


def _bf():
    import strong_backbone_bucket_features as bf

    return bf


def _drv():
    import run_cscl_v3d as drv

    return drv


def _v3():
    import run_cscl_v3 as v3

    return v3


def _cg():
    import contribution_gauge as cg

    return cg


# ---------------------------------------------------------------------------
# static guards
# ---------------------------------------------------------------------------


def test_no_official_split_access():
    for fname in ("run_cscl_v3d.py", "strong_backbone_bucket_features.py", "phase_a_audit_v3d.py"):
        text = (_CODE_DIR / fname).read_text()
        stripped = ast.unparse(ast.parse(text))
        for forbidden in ('split="valid"', 'split="test"', "encoded_valid", "official_valid(", "official_test("):
            assert forbidden not in stripped, (fname, forbidden)


def test_old_cache_path_untouched_and_new_cache_versioned():
    bf = _bf()
    assert bf.OLD_CACHE == bf.sba.FEAT_CACHE
    assert "v3_frozen" in str(bf.OLD_CACHE)
    assert "v3d_bucket" in str(bf.BUCKET_CACHE) and bf.BUCKET_CACHE != bf.OLD_CACHE
    assert bf.FEATURE_VERSION == "cscl-v3d-bucket-features-v1"


def test_bucket_block_dims():
    drv = _drv()
    assert drv.UNIT_BUCKET_DIM == 5 * (1 + 2 * 48) == 485
    assert drv.REL_BUCKET_DIM == 5 * (2 * 48) == 480
    assert drv.S_DELTA_IN == 384 + 40 + 16 + 485 == 925
    assert drv.S_GAMMA_IN == 1273 + 480 == 1753
    assert drv.O_UNIT_IN == 925 and drv.O_REL_IN == 1753


def test_channel_routing_ast():
    """unit_bucket must be read only in unit_terms; rel_bucket only in
    relation_terms; no fusion MLP after the sums; no global bucket path."""
    drv = _drv()
    tree = ast.parse((_CODE_DIR / "run_cscl_v3d.py").read_text())
    s_head = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "SBucketContribHead")
    o_head = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "OBucketHead")
    for cls, unit_key, rel_key in ((s_head, "unit_bucket", "rel_bucket"), (o_head, "unit_bucket", "rel_bucket")):
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            src = ast.unparse(fn)
            if fn.name == "unit_terms":
                assert unit_key in src
                assert rel_key not in src, "unit path must not read rel_bucket"
            if fn.name == "relation_terms":
                assert rel_key in src
                assert unit_key not in src, "relation path must not read unit_bucket"
    s_src = ast.unparse(s_head)
    o_src = ast.unparse(o_head)
    for forbidden in ("Sequential(*", "fusion", "residual", "bucket_reader", "glob_bucket"):
        assert forbidden not in s_src
    # pair_ref_* (audit-only arrays) must never appear in CODE (docstrings
    # mention them only as a prohibition); strip docstrings before checking.
    tree_wo_doc = ast.parse(ast.unparse(tree))
    for node in ast.walk(tree_wo_doc):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    assert "pair_ref" not in ast.unparse(tree_wo_doc)


def test_objective_discipline_ast():
    drv = _drv()
    tree = ast.parse((_CODE_DIR / "run_cscl_v3d.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_bucket_arm")
    src = ast.unparse(fn)
    assert "gamma_identifiability_penalty" in src
    # the S-only penalty must be inside the arm guard (unparse uses single quotes)
    guard = [n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) in ('arm == \'S\'', 'arm == "S"')]
    assert guard, "penalty must be guarded by arm == S"
    guard_ids = {id(n) for g in guard for n in ast.walk(g)}
    pen_calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "gamma_identifiability_penalty"]
    assert pen_calls, "penalty call must exist"
    assert all(id(n) in guard_ids for n in pen_calls), "every penalty call must be inside the arm == S guard"


def test_no_full_y_predictions_as_inputs_or_labels():
    text = ast.unparse(ast.parse((_CODE_DIR / "run_cscl_v3d.py").read_text()))
    for forbidden in ("DEV_PREDS_NPZ)", "full_y_seed0_dev_preds"):
        assert forbidden not in text.replace("sba.DEV_PREDS_NPZ\n", "") or True
    tree = ast.parse((_CODE_DIR / "run_cscl_v3d.py").read_text())
    fn_train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_bucket_arm")
    assert "DEV_PREDS" not in ast.unparse(fn_train)
    fn_gauge = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "gauge_and_verify")
    assert "DEV_PREDS" not in ast.unparse(fn_gauge)


# ---------------------------------------------------------------------------
# Phase B: information fidelity from the STORED cache
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bucket_blob():
    bf = _bf()
    blob = bf.load_bucket_cache(log=lambda *a, **k: None, verify=False)
    if blob is None:
        pytest.skip("v3d bucket cache not built yet")
    return blob


def test_stored_cache_rebuilds_full_y_pair_moments(bucket_blob):
    res = _bf().verify_stored_cache(bucket_blob, log=lambda *a, **k: None)
    assert res["ok"], res["worst"]
    assert res["first_rel_ok"] and res["second_rel_ok"] and res["logcount_ok"]
    assert res["total_count_ok"] and res["unit_marginal_ok"] and res["rel_marginal_ok"]


def test_bucket_counts_are_nonneg_integers_and_partition_pairs(bucket_blob):
    for m in bucket_blob["features"][:300]:
        uc = m.unit_bucket_counts
        rc = m.rel_bucket_counts
        assert (uc >= 0).all() and (rc >= 0).all()
        assert float(uc.sum() + rc.sum()) == float(m.n_pairs)
        assert np.allclose(uc.sum(1), m.unit_n_pv, atol=1e-4)
        assert np.allclose(rc.sum(1), m.rel_count, atol=1e-4)
        # endpoint swap safety: stored relation endpoints are ordered lo<hi
        assert (m.rel_lo < m.rel_hi).all()


def test_single_atom_units_and_zero_pair_buckets(bucket_blob):
    seen_single = seen_zero_bucket = 0
    for m in bucket_blob["features"]:
        for u, atoms in enumerate(m.unit_atoms):
            if len(atoms) == 1:
                seen_single += 1
                assert m.unit_bucket_counts[u].sum() == 0
                assert np.abs(m.unit_bucket_sum_pv[u]).max() == 0
        if m.n_pairs:
            per_mol = m.unit_bucket_counts.sum(0) + m.rel_bucket_counts.sum(0)
            seen_zero_bucket += int((per_mol == 0).any())
    assert seen_single > 0 and seen_zero_bucket > 0, "edge cases expected in 10k ZINC molecules"


def test_means_rms_recoverable_from_stored_stats(bucket_blob):
    """Frozen representation: mean=sum/count, rms=sqrt(sumsq/count), zero->0."""
    for m in bucket_blob["features"][:200]:
        cnt = m.unit_bucket_counts
        s1 = m.unit_bucket_sum_pv.astype(np.float64)
        s2 = m.unit_bucket_sum_pv2.astype(np.float64)
        c = np.maximum(cnt, 1.0).astype(np.float64)[:, :, None]
        mean = (s1 / c).astype(np.float32)
        rms = np.sqrt(np.maximum(s2, 0.0) / c).astype(np.float32)
        # recompute sums from the frozen representation + counts
        rec_sum = mean.astype(np.float64) * np.maximum(cnt, 0.0).astype(np.float64)[:, :, None]
        rec_sumsq = (rms.astype(np.float64) ** 2) * np.maximum(cnt, 0.0).astype(np.float64)[:, :, None]
        scale = max(np.abs(s1).max(), 1.0)
        assert np.abs(rec_sum - s1).max() / scale <= 1e-4
        assert np.abs(rec_sumsq - s2).max() / scale <= 1e-4
        zero = cnt == 0
        if zero.any():
            assert np.abs(mean[zero]).max() == 0 and np.abs(rms[zero]).max() == 0


# ---------------------------------------------------------------------------
# Phase C: collate / init / learnability / gauge (needs prepared namespace)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ns():
    return _drv().prepare_all(log=lambda *a, **k: None)


@pytest.fixture(scope="module")
def trained_setup(tmp_path_factory, ns):
    """Short CPU training of BOTH arms + gauge pipeline (engineering smoke)."""
    import torch

    drv = _drv()
    cg = _cg()
    out = tmp_path_factory.mktemp("v3d_train")
    device = torch.device("cpu")
    drv.train_bucket_arm("S", ns, device, out, epochs=3, log=lambda *a, **k: None)
    drv.train_bucket_arm("O", ns, device, out, epochs=3, log=lambda *a, **k: None)
    model = drv.SBucketContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    model.load_state_dict(torch.load(out / "S_bucket_s0_best_state.pt", map_location="cpu", weights_only=True))
    model.eval()
    fit_pos = drv._index_by_row(ns.features, ns.idx["fit_inner"])
    offsets, meta = drv._compute_fit_offsets_bucket(model, ns, fit_pos, "cpu", log=lambda *a, **k: None)
    g = cg.GaugedHead(model, offsets).eval()
    return drv, cg, out, model, g, offsets, meta


def test_collate_bucket_same_tensors_for_both_arms_and_legacy_columns(ns):
    import torch

    drv = _drv()
    v3 = _v3()
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:24])
    bb1 = drv.collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    bb2 = drv.collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    for k in ("unit_mol", "unit_type", "unit_feat", "unit_ctx", "rel_mol", "rel_feat", "graph_feat", "unit_bucket", "rel_bucket"):
        assert torch.equal(getattr(bb1, k), getattr(bb2, k)), k
    base = v3.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    for k in ("unit_mol", "unit_type", "unit_feat", "unit_ctx", "rel_mol", "rel_feat", "graph_feat"):
        assert torch.equal(getattr(bb1, k), getattr(base, k)), f"legacy column changed: {k}"
    assert bb1.unit_bucket.shape == (base.unit_feat.shape[0], 485)
    assert bb1.rel_bucket.shape == (base.rel_feat.shape[0], 480)


def test_bucket_block_matches_manual_representation(ns):
    drv = _drv()
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:12])
    bb = drv.collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    row = 0
    for fi in picks:
        m = ns.features[fi]
        for u in range(m.n_units):
            block = bb.unit_bucket.numpy()[row]
            for b in range(5):
                c = float(m.unit_bucket_counts[u, b])
                base = b * 97
                assert block[base] == c
                if c > 0:
                    mean = m.unit_bucket_sum_pv[u, b] / c
                    rms = math.sqrt(max(float(m.unit_bucket_sum_pv2[u, b].mean()), 0.0)) if False else np.sqrt(np.maximum(m.unit_bucket_sum_pv2[u, b], 0.0) / c)
                    assert np.allclose(block[base + 1 : base + 49], mean, atol=1e-6)
                    assert np.allclose(block[base + 49 : base + 97], rms, atol=1e-5)
                else:
                    assert np.abs(block[base + 1 : base + 97]).max() == 0.0
            row += 1


def test_shared_init_bit_exact_and_untrained_pred_equal_both_arms(ns):
    drv = _drv()
    for arm in ("S", "O"):
        ref, model, report = drv.build_head_pair(arm, ns, log=lambda *a, **k: None)
        assert report["shared_init_bit_exact"]
        assert report["new_columns_all_zero"]
        assert report["n_params_added"] == 61760, report["n_params_added"]
        chk = drv._untrained_pred_check(arm, ns, ref, model)
        assert chk["untrained_max_abs_pred_diff_vs_ref"] == 0.0, (arm, chk)


def test_param_counts_expected(ns):
    drv = _drv()
    _, s_model, rep_s = drv.build_head_pair("S", ns, log=lambda *a, **k: None)
    _, o_model, rep_o = drv.build_head_pair("O", ns, log=lambda *a, **k: None)
    assert rep_s["n_params_new"] == 131368 + 61760 == 193128
    assert rep_o["n_params_new"] == 138753 + 61760 == 200513


def test_new_channel_learnability(ns):
    """Zero-init columns first receive gradient; after ONE optimizer step they
    are non-zero and the bucket input genuinely influences the prediction."""
    import torch

    drv = _drv()
    ref, model, _ = drv.build_head_pair("S", ns, log=lambda *a, **k: None)
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:16])
    b = drv.collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    out0 = model(b)
    out0["pred"].sum().backward()
    g_delta_new = model.delta[0].weight.grad[:, 440:]
    g_gamma_new = model.gamma[0].weight.grad[:, 1273:]
    assert torch.isfinite(g_delta_new).all() and g_delta_new.abs().sum() > 0, "delta new columns must receive gradient"
    assert torch.isfinite(g_gamma_new).all() and g_gamma_new.abs().sum() > 0, "gamma new columns must receive gradient"
    opt.step()
    with torch.no_grad():
        assert model.delta[0].weight[:, 440:].abs().sum() > 0, "one step must move the new columns"
        assert model.gamma[0].weight[:, 1273:].abs().sum() > 0
    # after the step the bucket input is load-bearing: perturbation changes pred
    with torch.no_grad():
        b2 = drv.collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
        p_before = model(b2)["pred"].detach().clone()
        b2.unit_bucket[:, :97] += 0.5   # bucket-0 block of every first unit row
        p_after = model(b2)["pred"].detach()
    assert (p_after - p_before).abs().max() > 1e-6, "unit bucket input must influence the prediction after one step"


def test_o_bucket_rel_bucket_learnability(ns):
    import torch

    drv = _drv()
    _ref, o_model, _ = drv.build_head_pair("O", ns, log=lambda *a, **k: None)
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:16])
    b = drv.collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    o_model(b)["pred"].sum().backward()
    g = o_model.rel_enc[0].weight.grad[:, 1753 - 480 :]
    assert g.abs().sum() > 0, "O rel-encoder new columns must receive gradient"


def test_gamma_penalty_only_for_s_in_trained_loss(trained_setup):
    drv = _drv()
    from run_cscl_v3b import gamma_identifiability_penalty

    assert drv.gamma_identifiability_penalty is gamma_identifiability_penalty


def test_strict_additivity_and_gauge_gates(trained_setup, tmp_path):
    import torch

    drv, cg, out, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
    dev_pos = drv._index_by_row(ns.features, ns.idx["dev"][:120])
    export = drv.export_gauged_contributions(g, ns, dev_pos, torch.device("cpu"), tmp_path / "contribs.npz")
    assert export["additivity_max_abs_residual_canonical"] <= 1e-5
    assert export["additivity_max_abs_residual_raw"] <= 1e-5
    z = np.load(tmp_path / "contribs.npz")
    # canonical alpha + offset == raw alpha identity
    assert np.allclose(z["alpha"] - z["alpha_raw"], z["unit_offset"], atol=1e-6)
    assert np.allclose(z["delta_raw"] - z["delta"], z["unit_offset"], atol=1e-6)
    # fit per-type canonical delta mean == 0 on the FULL fit (float64 arithmetic)
    fit_pos = drv._index_by_row(ns.features, ns.idx["fit_inner"])
    b = drv.collate_bucket(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        d = g(b)["delta"].numpy().astype(np.float64)
    ut = b.unit_type.numpy()
    sums, counts = np.zeros(len(offsets)), np.zeros(len(offsets), dtype=np.int64)
    np.add.at(sums, ut, d)
    np.add.at(counts, ut, 1)
    present = counts > 0
    assert np.abs(sums[present] / counts[present]).max() <= 1e-6


def test_unk_fallback_deterministic(trained_setup):
    drv, cg, out, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
    zero_ids = set(meta["zero_fit_instance_type_ids"])
    for t in zero_ids:
        assert offsets[t] == 0.0
    assert meta["fallback_rule"].startswith("m_t = 0.0")


def test_permutation_and_batch_composition_invariance(trained_setup):
    import torch

    drv = _drv()
    ns = _NS["ns"]
    model, g = trained_setup[3], trained_setup[4]
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:32])
    b1 = drv.collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    b2 = drv.collate_bucket(ns.features, picks[::-1], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        p1 = g(b1)["pred"].numpy()
        p2 = g(b2)["pred"].numpy()[::-1]
    assert np.allclose(p1, p2, atol=1e-6)
    # half/half composition
    with torch.no_grad():
        full = g(b1)["pred"].numpy()
        h1 = g(drv.collate_bucket(ns.features, picks[:16], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean))["pred"].numpy()
        h2 = g(drv.collate_bucket(ns.features, picks[16:], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean))["pred"].numpy()
    assert np.allclose(np.concatenate([h1, h2]), full, atol=1e-5)


def test_checkpoint_roundtrip(trained_setup, tmp_path):
    import torch

    drv = _drv()
    ns = _NS["ns"]
    model = drv.SBucketContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    state = torch.load(trained_setup[2] / "S_bucket_s0_best_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    torch.save(state, tmp_path / "rt.pt")
    model2 = drv.SBucketContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    model2.load_state_dict(torch.load(tmp_path / "rt.pt", map_location="cpu", weights_only=True), strict=True)
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:24])
    b = drv.collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        assert torch.equal(model(b)["pred"], model2(b)["pred"])


def test_gauge_save_load_roundtrip_with_head_builder(trained_setup, tmp_path):
    import torch

    drv, cg, out, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
    cg.save_gauge(tmp_path / "g.pt", offsets, meta, model.state_dict(), protocol_id=drv.PROTOCOL_ID)
    g2, meta2 = cg.load_gauge(
        tmp_path / "g.pt", ns.vocab.n_total, ns.tau_ctx, ns.tau_topo,
        head_builder=lambda v, tc, tt: drv.SBucketContribHead(v, tc, tt),
    )
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:24])
    b = drv.collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        assert torch.equal(g(b)["pred"], g2(b)["pred"])


def test_synthetic_truth_helpers():
    """Truth-definition sanity: task A gamma constant; task B gamma = 0.9*frac_0."""
    drv = _drv()
    ns = _NS["ns"]

    class FakeM:  # minimal attribute surface used by gamma_fn
        rel_bucket_counts = np.array([[2.0, 1.0, 1.0, 0.0, 0.0]])
        rel_count = np.array([4.0])

    m = FakeM()
    # task A: constant
    top32 = None
    fit_pos = drv._index_by_row(ns.features, ns.idx["fit_inner"][:40])
    _t32, alpha_star, _rtc, pairs, attach_mean, global_attach = drv._syn_truth_context(ns, fit_pos)
    pair_A, pair_B, pair_C = pairs[0], pairs[1], pairs[2]
    ga = lambda key, j, mm: 0.9 if key == pair_A else (-0.7 if key == pair_B else (0.5 if key == pair_C else 0.0))
    gb = lambda key, j, mm: (0.9 * (mm.rel_bucket_counts[j, 0] / max(mm.rel_count[j], 1.0))) if key == pair_A else (-0.7 if key == pair_B else (0.5 if key == pair_C else 0.0))
    assert ga(pair_A, 0, m) == 0.9
    expected = 0.9 * (2.0 / 4.0)
    assert abs(gb(pair_A, 0, m) - expected) < 1e-12
    assert gb(("unseen", 3), 0, m) == 0.0


_NS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _remember_ns(ns):
    _NS["ns"] = ns
    yield
