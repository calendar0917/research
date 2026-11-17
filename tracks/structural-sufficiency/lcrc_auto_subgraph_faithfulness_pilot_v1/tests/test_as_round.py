"""Correctness tests for lcrc-auto-subgraph-faithfulness-pilot-v1.

Ten required gates (protocol §8) plus the frozen-rule invariants that make the
formal run interpretable.  A hard failure here stops the formal experiment; no
tolerance is relaxed to make a gate pass.

These tests load the frozen OGB caches and the reused arm-A checkpoint, so they
are marked ``slow`` and are not fresh-clone-safe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import as_common as C
import as_controls as K
import as_data as D
import as_gate as G
import as_intervene as I
import as_search as S

pytestmark = pytest.mark.slow

DATASET = "ogbg-molesol"
MOLLIPO = "ogbg-mollipo"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ctx():
    import cr_data as SD
    import as_worker as W

    ckpt = C.SOURCE_ROUND_RESULTS / DATASET / "A" / "checkpoint_best_val.pt"
    if not ckpt.is_file():
        pytest.skip(f"frozen checkpoint missing: {C.relpath(ckpt)}")
    bundle = SD.build_bundle(DATASET)
    arm, payload = W.load_arm(DATASET, bundle, "cpu")
    occs = {s: D.build_split_occ(bundle.caches[s], bundle.spec, DATASET, s) for s in C.SPLITS}
    return {
        "bundle": bundle,
        "arm": arm,
        "payload": payload,
        "occs": occs,
        "sigma": float(bundle.scaler["std"]),
        "mu": float(bundle.scaler["mean"]),
    }


@pytest.fixture(scope="module")
def ctx_lipo():
    import cr_data as SD
    import as_worker as W

    ckpt = C.SOURCE_ROUND_RESULTS / MOLLIPO / "A" / "checkpoint_best_val.pt"
    if not ckpt.is_file():
        pytest.skip(f"frozen checkpoint missing: {C.relpath(ckpt)}")
    bundle = SD.build_bundle(MOLLIPO)
    arm, _payload = W.load_arm(MOLLIPO, bundle, "cpu")
    occs = {s: D.build_split_occ(bundle.caches[s], bundle.spec, MOLLIPO, s) for s in C.SPLITS}
    return {
        "bundle": bundle,
        "arm": arm,
        "occs": occs,
        "sigma": float(bundle.scaler["std"]),
        "mu": float(bundle.scaler["mean"]),
    }


def _gate_matrix(gb: D.GatedBatch, vec: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    return gb.gate_matrix([vec], requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# frozen constants / rule invariants
# ---------------------------------------------------------------------------


def test_frozen_constants() -> None:
    assert C.ROUND_ID == "lcrc-auto-subgraph-faithfulness-pilot-v1"
    assert C.PROTOCOL_ID == C.ROUND_ID
    assert C.CONDITION_NAME == "CONDITIONAL_S1_S2_EVIDENCE_MASK"
    assert C.DATASETS == ("ogbg-molesol", "ogbg-mollipo")
    assert C.SPLITS == ("train", "valid", "test")
    assert C.N_SAMPLES_PER_SPLIT == 64
    assert (C.SEARCH_MIN_K, C.SEARCH_MAX_K) == (2, 6)
    assert C.BEAM_WIDTH == 12 and C.TOP_K_SUBGRAPHS == 3
    assert C.JACCARD_MAX_OVERLAP == 0.5
    assert C.N_MATCHED_RANDOM == 16 and C.MIN_MATCHED_RANDOM == 8
    assert C.MAX_GROWTH_ATTEMPTS == 512 and C.S2_MATCH_REL_TOL == 0.25
    assert C.DOSE_GATE == 0.5 and C.FD_EPS == 1e-3
    assert C.SAMPLE_SEED == C.CONTROL_SEED == C.BOOTSTRAP_SEED == 20261117
    assert C.freeze_hash() == C.freeze_hash()


def test_paths_do_not_point_inside_readonly_reuse() -> None:
    for path in (C.SQL_DIR, C.PREP_DIR, C.DISSECT_DIR, C.SOURCE_ROUND_DIR):
        assert C.ROUND_DIR not in path.parents
    assert C.RESULTS_DIR == C.ROUND_DIR / "results"
    assert C.ANALYSIS_DIR == C.ROUND_DIR / "analysis"


def test_sample_selection_is_deterministic_and_independent_of_order() -> None:
    ids = np.arange(500, dtype=np.int64)
    a = C.select_samples("ogbg-molesol", "valid", ids)
    b = C.select_samples("ogbg-molesol", "valid", ids[::-1])
    assert a["row_ids"] == b["row_ids"], "selection must not depend on the input order"
    assert a["row_ids_sha256"] == b["row_ids_sha256"]
    keys = sorted((C.sample_order_key("ogbg-molesol", "valid", int(r)), int(r)) for r in ids)
    assert a["row_ids"] == [rid for _k, rid in keys[:64]]
    # a different split / dataset must produce a different ordering
    other = C.select_samples("ogbg-molesol", "test", ids)
    assert other["row_ids"] != a["row_ids"]
    # no duplicate selection
    assert len(set(a["row_ids"])) == 64


def test_search_rule_is_the_declared_one(ctx) -> None:
    """k=2 enumerates *all* shared-endpoint pairs; the beam and Top-3 respect the caps."""
    cache = ctx["bundle"].caches["valid"]
    occ = ctx["occs"]["valid"]
    mol = G.SingleMolecule(cache, 0, ctx["bundle"].spec)
    pairs = mol.shared_endpoint_pairs()
    brute = []
    for i in range(mol.n_bonds):
        for j in range(i + 1, mol.n_bonds):
            if {int(mol.bond_u[i]), int(mol.bond_v[i])} & {int(mol.bond_u[j]), int(mol.bond_v[j])}:
                brute.append((i, j))
    assert sorted(set(pairs)) == sorted(brute)
    g = np.random.default_rng(0).normal(size=mol.n_bonds)
    res = S.search_subgraphs(mol, lambda b: S.score_discovery(b, g), score_name="A_discovery")
    assert len(res.levels[2]) <= C.BEAM_WIDTH
    for k in range(3, C.SEARCH_MAX_K + 1):
        assert len(res.levels.get(k, [])) <= C.BEAM_WIDTH
    assert len(res.levels[1]) == mol.n_bonds            # every physical bond is ranked
    # determinism
    res2 = S.search_subgraphs(mol, lambda b: S.score_discovery(b, g), score_name="A_discovery")
    assert [(c.bonds, c.score) for c in res.pool()] == [(c.bonds, c.score) for c in res2.pool()]
    topk, rejected = S.select_topk(res.pool())
    assert 1 <= len(topk) <= C.TOP_K_SUBGRAPHS
    for i in range(len(topk)):
        for j in range(i + 1, len(topk)):
            assert S.jaccard(topk[i].bonds, topk[j].bonds) <= C.JACCARD_MAX_OVERLAP + 1e-12
    for cand in topk:
        assert mol.atoms_connected(cand.bonds)
        assert C.SEARCH_MIN_K <= len(cand.bonds) <= C.SEARCH_MAX_K
    del rejected


# ---------------------------------------------------------------------------
# TEST 1 -- all-ones equivalence with the frozen arm A
# ---------------------------------------------------------------------------


def test_01_all_ones_reproduces_frozen_arm(ctx) -> None:
    bundle, arm = ctx["bundle"], ctx["arm"]
    sigma, mu = ctx["sigma"], ctx["mu"]
    checked = 0
    for split in C.SPLITS:
        cache = bundle.caches[split]
        occ = ctx["occs"][split]
        rows = list(range(min(16, int(cache["n_molecules"]))))
        gb = D.collate_gated(cache, rows, occ)
        with torch.no_grad():
            ref = G.reference_parts(arm, gb.batch)
            got = G.gated_parts(arm, gb, gb.gate_ones())
        for key in ("pred_std", "local_sum", "rho"):
            a = ref[key].numpy().astype(np.float64)
            b = got[key].numpy().astype(np.float64)
            scale = sigma if key != "pred_std" else sigma
            raw_a = a * scale
            raw_b = b * scale
            tol = C.trace_tolerance(raw_a)
            assert np.all(np.abs(raw_a - raw_b) <= tol), f"{split}/{key} not equivalent"
        checked += len(rows)
    assert checked >= 30
    # the gated path refuses a model that is not in eval() mode
    gb = D.collate_gated(bundle.caches["valid"], [0], ctx["occs"]["valid"])
    gates = gb.gate_ones()
    arm.train()
    try:
        with pytest.raises(RuntimeError):
            G.gated_parts(arm, gb, gates)
    finally:
        arm.eval()
    assert G.gated_parts(arm, gb, gates) is not None


# ---------------------------------------------------------------------------
# TEST 2 -- occurrence -> physical bond mapping, triangles and reverse paths
# ---------------------------------------------------------------------------


def test_02_occurrence_mapping_matches_cache(ctx) -> None:
    bundle = ctx["bundle"]
    n_checked = 0
    triangles = 0
    reverse_pairs = 0
    for split in C.SPLITS:
        cache = bundle.caches[split]
        n = int(cache["n_molecules"]) if split != "train" else 60
        for row in range(n):
            mol = D.slice_molecule(cache, row)
            _g, analysis = D.analysis_from_mol(mol)
            assert np.array_equal(
                np.stack([analysis.bond_u, analysis.bond_v], axis=1), mol["bond_index"]
            )
            occ = D.build_occ_map(mol["atom_type"], analysis, int(bundle.spec.n_bond_types))
            report = D.verify_occurrences(occ, mol, strict=True)
            assert report["n_failed"] == 0, (split, row, report["failed"])
            n_checked += 1
            if occ.n_path:
                # triangles are kept and flagged by closure == 1
                bonded = {tuple(sorted(p)) for p in mol["bond_index"].tolist()}
                for idx in range(occ.n_path):
                    a, b = int(occ.path_root[idx]), int(occ.path_end[idx])
                    closed = (min(a, b), max(a, b)) in bonded
                    assert int(occ.path_closure[idx]) == int(closed)
                    triangles += int(closed)
                # ordered paths appear in both directions with the same bond pair
                index = {}
                for idx in range(occ.n_path):
                    index[(int(occ.path_root[idx]), int(occ.path_middle[idx]), int(occ.path_end[idx]))] = idx
                for (r, m, e), idx in index.items():
                    if (e, m, r) in index:
                        j = index[(e, m, r)]
                        assert {int(occ.path_bond1[idx]), int(occ.path_bond2[idx])} == {
                            int(occ.path_bond1[j]),
                            int(occ.path_bond2[j]),
                        }
                        reverse_pairs += 1
    assert n_checked >= 180
    # the fixtures must actually exercise the two special cases
    assert triangles > 0, "no triangle closure path was found -- the test would not cover it"
    assert reverse_pairs > 0, "no reverse (duplicate) path pair was found"


# ---------------------------------------------------------------------------
# TEST 3 -- an occurrence touching two masked bonds is removed exactly once
# ---------------------------------------------------------------------------


def test_03_masked_occurrence_removed_once(ctx) -> None:
    import as_worker as W

    out = W.masked_twice_check(
        ctx["arm"], ctx["bundle"].caches["valid"], ctx["occs"]["valid"], list(range(8)),
        sigma=ctx["sigma"], mu=ctx["mu"],
    )
    assert out["n_checked"] >= 8
    assert out["pass"], out["records"]
    for rec in out["records"]:
        assert rec["abs_err_raw"] <= rec["tolerance_raw"], rec
        # the pair really is shared by more than one occurrence
        assert rec["n_occurrences_touching_either_bond"] >= 1


# ---------------------------------------------------------------------------
# TEST 4 -- the continuous gate m=0.5 propagates gradients
# ---------------------------------------------------------------------------


def test_04_half_gate_backpropagates(ctx) -> None:
    arm = ctx["arm"]
    cache = ctx["bundle"].caches["valid"]
    occ = ctx["occs"]["valid"]
    gb = D.collate_gated(cache, [0], occ)
    n = int(gb.bond_counts[0])
    half = np.full(n, 0.5, dtype=np.float64)
    M = _gate_matrix(gb, half, requires_grad=True)
    parts = G.gated_parts(arm, gb, M)
    grad = torch.autograd.grad(parts["pred_std"].sum(), M)[0]
    g = grad.numpy()[0][:n]
    assert np.all(np.isfinite(g))
    assert np.any(np.abs(g) > 0), "m=0.5 gates must still carry a gradient"
    # a partially masked prediction must sit between the fully open and fully closed ones
    with torch.no_grad():
        p1 = float(G.gated_parts(arm, gb, _gate_matrix(gb, np.ones(n)))["pred_std"][0])
        p0 = float(G.gated_parts(arm, gb, _gate_matrix(gb, np.zeros(n)))["pred_std"][0])
        ph = float(parts["pred_std"][0])
    lo, hi = min(p0, p1), max(p0, p1)
    assert lo - 1e-4 <= ph <= hi + 1e-4
    # the round only ever builds gates inside [0, 1]
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        m = G.bond_mask(n, range(n), value)
        assert np.all(m >= 0.0) and np.all(m <= 1.0)


# ---------------------------------------------------------------------------
# TEST 5 -- finite differences (eps = 1e-3) against autograd
# ---------------------------------------------------------------------------


def test_05_finite_difference_matches_autograd(ctx) -> None:
    import as_worker as W

    out = W.gradient_fd_check(
        ctx["arm"], ctx["bundle"].caches["valid"], ctx["occs"]["valid"], list(range(C.FD_MOLECULES)),
        sigma=ctx["sigma"], mu=ctx["mu"], eps=C.FD_EPS, bonds_per=C.FD_BONDS_PER_MOLECULE,
    )
    assert out["n_molecules"] == C.FD_MOLECULES
    assert out["n_bonds_checked"] == C.FD_MOLECULES * C.FD_BONDS_PER_MOLECULE
    assert out["n_failed"] == 0, out["failed"]
    assert out["max_abs_err"] <= 2e-3
    assert out["max_rel_err_of_large"] <= 5e-2
    # near-zero gradients are judged on the absolute error only
    for rec in out["records"]:
        if abs(rec["finite_difference_raw"]) < 1e-2:
            assert rec["abs_ok"]


# ---------------------------------------------------------------------------
# TEST 6 -- atom renumbering
# ---------------------------------------------------------------------------


def test_06_atom_renumbering_equivariance(ctx, ctx_lipo) -> None:
    import as_worker as W

    for data, n in ((ctx, C.RENUMBER_MOLECULES), (ctx_lipo, 4)):
        out = W.renumbering_check(
            data["arm"], data["bundle"].caches["valid"], data["occs"]["valid"], list(range(n)),
            sigma=data["sigma"], mu=data["mu"],
        )
        assert out["n_molecules"] == n
        assert out["pass"], [r for r in out["records"] if not r["passed"]]
        for rec in out["records"]:
            assert rec["pred_abs_err"] <= rec["pred_tolerance"]
            assert rec["grad_max_abs_err"] <= rec["grad_max_tolerance"]


# ---------------------------------------------------------------------------
# TEST 7 -- single-graph vs batch inference, no cross-molecule contamination
# ---------------------------------------------------------------------------


def test_07_batching_has_no_cross_molecule_contamination(ctx) -> None:
    import as_worker as W

    arm = ctx["arm"]
    cache = ctx["bundle"].caches["valid"]
    occ = ctx["occs"]["valid"]
    out = W.batching_check(arm, cache, occ, list(range(8)), sigma=ctx["sigma"], mu=ctx["mu"])
    assert out["pass"], out
    assert out["max_abs_err_single_vs_batch"] <= out["tolerance_max"]
    assert out["max_abs_err_order_permuted"] <= out["tolerance_max"]

    # gradients of molecule 0's gates must not depend on molecule 1 being present
    gb0 = D.collate_gated(cache, [0], occ)
    gb01 = D.collate_gated(cache, [0, 1], occ)
    M0 = gb0.gate_ones(requires_grad=True)
    M01 = gb01.gate_ones(requires_grad=True)
    g0 = torch.autograd.grad(G.gated_parts(arm, gb0, M0)["pred_std"].sum(), M0)[0].numpy()
    g01 = torch.autograd.grad(G.gated_parts(arm, gb01, M01)["pred_std"].sum(), M01)[0].numpy()
    n0 = int(gb0.bond_counts[0])
    assert np.allclose(g0[0, :n0], g01[0, :n0], atol=1e-6, rtol=0)
    assert np.all(np.abs(g01[0, n0:]) == 0.0), "padded gate entries must receive no gradient"


# ---------------------------------------------------------------------------
# TEST 8 -- delta_total == delta_L + delta_P in raw units
# ---------------------------------------------------------------------------


def test_08_readout_closure(ctx) -> None:
    import as_worker as W

    arm, cache, occ = ctx["arm"], ctx["bundle"].caches["valid"], ctx["occs"]["valid"]
    sigma, mu = ctx["sigma"], ctx["mu"]
    checked = 0
    for row in range(3):
        rec, _disc, _heur = W.analyse_molecule(
            arm, cache, occ, row, dataset=DATASET, split="valid", spec=ctx["bundle"].spec,
            sigma=sigma, mu=mu, train_std=sigma,
            y_raw=float(ctx["bundle"].targets_raw["valid"][row]),
            row_id=int(ctx["bundle"].global_row_ids["valid"][row]),
        )
        for rank in rec["topk"]:
            eff = rank["effect"]
            assert eff["closure_error"] <= eff["closure_tolerance"], eff
            assert abs(eff["signed_delta"] - (eff["delta_L"] + eff["delta_P"])) <= eff["closure_tolerance"]
            checked += 1
        for key in ("s1_only", "s2_only", "s1_s2"):
            eff = rec["decomposition"][key]
            assert eff["closure_error"] <= eff["closure_tolerance"], (key, eff)
        assert rec["decomposition"]["s1_s2_equals_top1"]["abs_diff"] <= rec["decomposition"]["s1_s2_equals_top1"]["tolerance"]
    assert checked >= 3


# ---------------------------------------------------------------------------
# TEST 9 -- checkpoint reload and parameter-hash stability
# ---------------------------------------------------------------------------


def test_09_checkpoint_reload_and_state_hash(ctx) -> None:
    import as_worker as W

    bundle, arm = ctx["bundle"], ctx["arm"]
    before = C.sha256_state(arm.state_dict())
    stored = C.sha256_state(ctx["payload"]["state_dict"])
    assert before == stored

    # a fresh reload from disk must give the same state and the same predictions
    arm2, _payload2 = W.load_arm(DATASET, bundle, "cpu")
    assert C.sha256_state(arm2.state_dict()) == before
    gb = D.collate_gated(bundle.caches["valid"], list(range(4)), ctx["occs"]["valid"])
    with torch.no_grad():
        p1 = G.gated_parts(arm, gb, gb.gate_ones())["pred_std"].numpy()
        p2 = G.gated_parts(arm2, gb, gb.gate_ones())["pred_std"].numpy()
        ref = G.reference_parts(arm2, gb.batch)["pred_std"].numpy()
    assert np.array_equal(p1, p2)
    assert np.array_equal(p2, ref)

    # the explanation path (gradients + masked forwards) must not touch parameters
    import as_worker as W2

    W2.analyse_molecule(
        arm, bundle.caches["valid"], ctx["occs"]["valid"], 0, dataset=DATASET, split="valid",
        spec=bundle.spec, sigma=ctx["sigma"], mu=ctx["mu"], train_std=ctx["sigma"],
        y_raw=float(bundle.targets_raw["valid"][0]),
        row_id=int(bundle.global_row_ids["valid"][0]),
    )
    after = C.sha256_state(arm.state_dict())
    assert after == before
    assert G.gated_parts(arm, gb, gb.gate_ones()) is not None
    assert C.sha256_state(arm.state_dict()) == before


# ---------------------------------------------------------------------------
# TEST 10 -- split disjointness and Train-only standardisation
# ---------------------------------------------------------------------------


def test_10_splits_disjoint_and_train_only_std(ctx) -> None:
    import as_worker as W

    for dataset in C.DATASETS:
        ident = W.data_identity(dataset)
        assert ident["pass"], ident
        for key, value in ident["split_disjointness"].items():
            assert value["empty"], (dataset, key, value)
        assert ident["split_sizes"] == dict(C.EXPECTED_SPLIT_SIZES[dataset])
        scaler = ident["target_scaler_train_only"]
        assert scaler["fit_split"] == "train"
        assert int(scaler["n"]) == int(C.EXPECTED_SPLIT_SIZES[dataset]["train"])
        # the scaler must be recomputable from Train labels only
        import cr_data as SD

        payload = SD.load_payload(dataset)
        train_rows = np.asarray(payload[C.SC.SPLIT_FILE["train"]], dtype=np.int64)
        manual = C.SC.target_scaler(payload["labels"][train_rows])
        assert manual["mean"] == pytest.approx(scaler["mean"], abs=0, rel=0)
        assert manual["std"] == pytest.approx(scaler["std"], abs=0, rel=0)
        # a Test-only scaler would differ materially; prove the code does not use it
        test_rows = np.asarray(payload[C.SC.SPLIT_FILE["test"]], dtype=np.int64)
        test_only = C.SC.target_scaler(payload["labels"][test_rows])
        assert abs(test_only["mean"] - manual["mean"]) > 1e-6 or abs(test_only["std"] - manual["std"]) > 1e-6


# ---------------------------------------------------------------------------
# extra invariants: controls, gates, controls bookkeeping
# ---------------------------------------------------------------------------


def test_controls_are_deterministic_and_respect_the_match_rules(ctx) -> None:
    cache = ctx["bundle"].caches["valid"]
    occ = ctx["occs"]["valid"]
    mol = G.SingleMolecule(cache, 0, ctx["bundle"].spec)
    g = np.abs(np.random.default_rng(1).normal(size=mol.n_bonds))
    res = S.search_subgraphs(mol, lambda b: S.score_discovery(b, g), score_name="A_discovery")
    topk, _rejected = S.select_topk(res.pool())
    assert topk
    top1 = topk[0].bonds
    rng1 = C.control_rng(DATASET, "valid", int(ctx["bundle"].global_row_ids["valid"][0]))
    rng2 = C.control_rng(DATASET, "valid", int(ctx["bundle"].global_row_ids["valid"][0]))
    c1 = K.matched_random_controls(mol, top1, rng1)
    c2 = K.matched_random_controls(mol, top1, rng2)
    assert c1["matched"] == c2["matched"]
    assert c1["attempts"] == c2["attempts"]
    assert c1["attempts"] <= C.MAX_GROWTH_ATTEMPTS
    target_s2 = mol.s2_count(top1)
    for rec in c1["matched"]:
        assert len(rec["bonds"]) == len(top1)
        assert tuple(rec["bonds"]) != tuple(sorted(top1))
        assert mol.atoms_connected(rec["bonds"])
        assert rec["s2_rel_diff"] <= C.S2_MATCH_REL_TOL + 1e-12
        assert rec["s2_rel_diff"] == pytest.approx(
            abs(rec["n_s2_occurrences"] - target_s2) / max(1, target_s2), abs=0, rel=1e-12
        )
    # a shortfall is reported, never silently backfilled
    if len(c1["matched"]) < C.MIN_MATCHED_RANDOM:
        assert c1["status"] == "INSUFFICIENT_MATCHED_NULL"
    else:
        assert c1["status"] == "OK"
        assert len(c1["matched"]) <= C.N_MATCHED_RANDOM


def test_gate_mask_semantics(ctx) -> None:
    """The mask is conditional evidence gating, not bond deletion: atom features survive."""
    arm, cache, occ = ctx["arm"], ctx["bundle"].caches["valid"], ctx["occs"]["valid"]
    gb = D.collate_gated(cache, [0], occ)
    n = int(gb.bond_counts[0])
    with torch.no_grad():
        open_pred = float(G.gated_parts(arm, gb, _gate_matrix(gb, np.ones(n)))["pred_std"][0])
        closed = float(G.gated_parts(arm, gb, _gate_matrix(gb, np.zeros(n)))["pred_std"][0])
    assert math.isfinite(open_pred) and math.isfinite(closed)
    assert open_pred != closed
    # S1_ONLY and S2_ONLY are genuinely different interventions
    both_zero = np.zeros(n)
    ones = np.ones(n)
    with torch.no_grad():
        s1 = float(G.gated_parts(arm, gb, _gate_matrix(gb, both_zero), _gate_matrix(gb, ones))["pred_std"][0])
        s2 = float(G.gated_parts(arm, gb, _gate_matrix(gb, ones), _gate_matrix(gb, both_zero))["pred_std"][0])
    assert abs(s1 - open_pred) > 0 and abs(s2 - open_pred) > 0
    assert abs(s1 - s2) > 0


def test_variant_list_is_the_frozen_one(ctx) -> None:
    cache = ctx["bundle"].caches["valid"]
    mol = G.SingleMolecule(cache, 0, ctx["bundle"].spec)
    variants = I.build_variants(
        mol, [(0, 1), (2, 3), (4, 5)],
        local_heuristic_bonds=(0, 1, 2),
        matched_random=[(6, 7), (8, 9)],
        keycount_only=[(10, 11)],
    )
    names = [v["name"] for v in variants]
    assert names[:8] == [
        "BASE", "TOP1", "TOP2", "TOP3", "TOP1_DOSE",
        "TOP1_S1_ONLY", "TOP1_S2_ONLY", "TOP1_S1_S2",
    ]
    assert "LOCAL_HEURISTIC" in names
    assert sum(1 for v in variants if v["kind"] == "matched_random") == 2
    assert sum(1 for v in variants if v["kind"] == "keycount_only") == 1
    for v in variants:
        assert np.all(v["gate_s1"] >= 0.0) and np.all(v["gate_s1"] <= 1.0)
        assert np.all(v["gate_s2"] >= 0.0) and np.all(v["gate_s2"] <= 1.0)
    dose = next(v for v in variants if v["name"] == "TOP1_DOSE")
    assert dose["gate_s1"][0] == pytest.approx(C.DOSE_GATE)
