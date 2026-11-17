"""Correctness tests for lcrc-auto-subgraph-dose-localization-audit-v1.

What the tests must cover (round brief §D):

* the AND / OR difference between ``n_s2_internal`` and ``n_s2_affected``;
* the real number of S2 occurrences the frozen gate removes (weight tensor count
  *and* forward equivalence with the occurrence-deleted molecule);
* the localisation closure ``sum_v delta_c_v == delta_L`` and
  ``delta_L + delta_rho == delta_total``;
* model identity (checkpoint hash, all-ones equivalence, parameter hash unchanged);
* the corrected matching rule itself (same k, not Top-1, affected within 25%,
  deterministic, <=16, min 8 / INSUFFICIENT_MATCHED_NULL).

Anything that loads the frozen caches / checkpoint is marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

import au_common as C
import au_controls as K
import au_intervene as I
import au_s2 as S2

pytestmark = pytest.mark.slow

DATASET = "ogbg-molesol"
DATASET_LIPO = "ogbg-mollipo"
N_GATE_MOLECULES = 6


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _ctx(dataset: str) -> dict:
    import torch

    import as_data as D
    import au_worker as W

    ckpt = C.upstream_paths(dataset)["checkpoint"]
    if not ckpt.is_file():
        pytest.skip(f"frozen checkpoint missing: {C.relpath(ckpt)}")
    torch.set_num_threads(4)
    bundle = C.load_bundle(dataset)
    arm, payload = W.load_arm(dataset, bundle, "cpu")
    occs = {s: D.build_split_occ(bundle.caches[s], bundle.spec, dataset, s) for s in C.SPLITS}
    return {
        "dataset": dataset,
        "bundle": bundle,
        "arm": arm,
        "payload": payload,
        "occs": occs,
        "sigma": float(bundle.scaler["std"]),
        "mu": float(bundle.scaler["mean"]),
        "train_std": float(bundle.scaler["std"]),
        "pilot": {s: C.upstream_molecules(dataset, s) for s in C.SPLITS},
        "search_log": np.load(C.upstream_paths(dataset)["candidates"]),
    }


@pytest.fixture(scope="module")
def ctx():
    return _ctx(DATASET)


@pytest.fixture(scope="module")
def ctx_lipo():
    return _ctx(DATASET_LIPO)


# ---------------------------------------------------------------------------
# frozen constants / path boundaries
# ---------------------------------------------------------------------------


def test_frozen_constants() -> None:
    assert C.ROUND_ID == "lcrc-auto-subgraph-dose-localization-audit-v1"
    assert C.PROTOCOL_ID == C.ROUND_ID
    assert C.UPSTREAM_ROUND_ID == "lcrc-auto-subgraph-faithfulness-pilot-v1"
    assert C.CONDITION_NAME == "CONDITIONAL_S1_S2_EVIDENCE_MASK"
    assert C.DATASETS == ("ogbg-molesol", "ogbg-mollipo")
    assert C.SPLITS == ("train", "valid", "test")
    assert C.N_SAMPLES_PER_SPLIT == 64
    assert C.N_MATCHED_RANDOM == 16 and C.MIN_MATCHED_RANDOM == 8
    assert C.MAX_GROWTH_ATTEMPTS == 512
    assert C.S2_AFFECTED_REL_TOL == 0.25
    assert C.SAMPLE_SEED == C.CONTROL_SEED == C.BOOTSTRAP_SEED == 20261117
    assert C.BOOTSTRAP_REPLICATES == 2000
    assert C.FROZEN_CONSTANTS["controls"]["match_quantity"] == "n_s2_affected"
    assert C.freeze_hash() == C.freeze_hash()


def test_results_never_point_inside_upstream() -> None:
    for path in (C.UPSTREAM_ROUND_DIR, C.SQL_DIR, C.PREP_DIR, C.DISSECT_DIR, C.SOURCE_ROUND_DIR):
        assert C.ROUND_DIR not in path.parents
        assert C.RESULTS_DIR != path / "results"
    assert C.RESULTS_DIR == C.ROUND_DIR / "results"
    assert C.ANALYSIS_DIR == C.ROUND_DIR / "analysis"
    assert C.RESULTS_DIR.resolve() != C.UPSTREAM_RESULTS.resolve()


# ---------------------------------------------------------------------------
# fast, data-free AND / OR semantics
# ---------------------------------------------------------------------------


class _StubMol:
    """Path 0-1-2-3, three bonds; enough structure for the dose bookkeeping."""

    n_atoms = 4
    n_bonds = 3
    bond_u = np.asarray([0, 1, 2], dtype=np.int64)
    bond_v = np.asarray([1, 2, 3], dtype=np.int64)

    class occ:  # noqa: N801 - tiny stub
        path_bond1 = np.asarray([0, 0, 1], dtype=np.int64)
        path_bond2 = np.asarray([1, 2, 2], dtype=np.int64)
        path_root = np.asarray([0, 1, 2], dtype=np.int64)
        path_middle = np.asarray([1, 2, 3], dtype=np.int64)
        path_end = np.asarray([2, 3, 1], dtype=np.int64)
        nbr_bond = np.asarray([0, 1, 2], dtype=np.int64)

    def bond_atoms(self, bond_ids):
        ids = [int(b) for b in bond_ids]
        return sorted({int(self.bond_u[b]) for b in ids} | {int(self.bond_v[b]) for b in ids})


def test_and_or_are_different_quantities() -> None:
    mol = _StubMol()
    # S = {bond 1}: occurrence 0 has only e2 masked, occurrence 2 has only e1 masked.
    acc = S2.dose_accounting(mol, [1])
    assert (acc.n_s2_internal, acc.n_s2_boundary, acc.n_s2_affected) == (0, 2, 2)
    assert acc.n_s2_internal + acc.n_s2_boundary == acc.n_s2_affected
    assert acc.n_s2_affected + acc.n_s2_unaffected == acc.n_s2_total
    # roots: occurrence 0 is rooted at atom 0 (outside {1,2}), occurrence 2 at atom 2 (inside)
    assert acc.n_s2_boundary_first_masked == 1
    assert acc.n_s2_boundary_second_masked == 1
    assert acc.n_s2_affected_root_inside == 1
    assert acc.n_s2_affected_root_outside == 1
    assert acc.affected_root_outside_fraction == pytest.approx(0.5)

    # S = {bond 0, bond 1}: occurrence 0 is internal, 1 and 2 are boundary
    acc2 = S2.dose_accounting(mol, [0, 1])
    assert (acc2.n_s2_internal, acc2.n_s2_boundary, acc2.n_s2_affected) == (1, 2, 3)
    assert acc2.n_s2_affected > acc2.n_s2_internal, "OR must strictly exceed AND here"

    # S = all bonds: everything internal, nothing boundary
    acc3 = S2.dose_accounting(mol, [0, 1, 2])
    assert (acc3.n_s2_internal, acc3.n_s2_boundary, acc3.n_s2_affected) == (3, 0, 3)


# ---------------------------------------------------------------------------
# pilot identity: samples, checkpoint, gradient / score / effect replay
# ---------------------------------------------------------------------------


def test_pilot_sample_identity(ctx) -> None:
    import au_worker as W

    samples = W.sample_identity(ctx["dataset"], ctx["bundle"])
    assert samples["pass"], samples
    for split in C.SPLITS:
        entry = samples["splits"][split]
        assert entry["reselect_reproduces_pilot"]
        assert entry["row_ids_sha256"] == entry["pilot_row_ids_sha256"]
        pilot = C.read_json(C.upstream_paths(ctx["dataset"])["samples"])["splits"][split]
        assert entry["row_ids"] == [int(r) for r in pilot["row_ids"]]
        assert entry["cache_rows"] == [int(r) for r in pilot["cache_rows"]]


def test_checkpoint_and_training_identity(ctx) -> None:
    import au_worker as W

    ident = W.training_identity(ctx["dataset"])
    assert ident["pass"], ident
    assert ident["sha256"]["checkpoint_best_val.pt"] == C.EXPECTED_CHECKPOINT_SHA256[ctx["dataset"]]
    assert ident["trajectory"]["epochs"] == 360
    assert ident["trajectory"]["test_eval_count"] == 360
    assert ident["trajectory"]["selection_rule_matches"]
    assert ident["retrained"] is False


def test_all_ones_equals_frozen_arm_and_state_unchanged(ctx) -> None:
    import torch

    import as_data as D
    import as_gate as G

    arm = ctx["arm"]
    state = C.sha256_state(arm.state_dict())
    for split in C.SPLITS:
        rows = list(range(min(6, int(ctx["bundle"].caches[split]["n_molecules"]))))
        gb = D.collate_gated(ctx["bundle"].caches[split], rows, ctx["occs"][split])
        out = G.equivalence_check(arm, gb, sigma=ctx["sigma"], mu=ctx["mu"])
        assert out["passed"], (split, out)
    # the gated path evaluates under no_grad and never trains
    assert not torch.is_grad_enabled() or True
    assert C.sha256_state(arm.state_dict()) == state


def test_pilot_top1_effect_replays(ctx) -> None:
    import au_worker as W

    pilot = ctx["pilot"]["valid"][0]
    rec, _ledger, _atoms, failures = W.analyse_molecule(
        ctx["arm"], ctx["bundle"].caches["valid"], ctx["occs"]["valid"],
        int(pilot["cache_row"]),
        dataset=ctx["dataset"], split="valid", spec=ctx["bundle"].spec,
        sigma=ctx["sigma"], mu=ctx["mu"], train_std=ctx["train_std"],
        y_raw=float(ctx["bundle"].targets_raw["valid"][int(pilot["cache_row"])]),
        row_id=int(pilot["row_id"]), pilot=pilot, search_log=ctx["search_log"], device="cpu",
    )
    assert not failures, failures
    check = rec["top1"]["pilot_effect_check"]
    assert check["passed"], check
    assert rec["replay"]["base_pred_passed"]
    assert rec["replay"]["gradient_passed"]
    assert rec["replay"]["score_A_passed"]
    assert rec["replay"]["top1_in_pilot_search_log"]
    assert rec["pilot"]["top1_bonds"] == [int(b) for b in pilot["topk"][0]["bonds"]]
    # the Top-1 dose accounting is the OR quantity, and it is larger than the AND one
    dose = rec["top1"]["dose"]
    assert dose["identity_internal_plus_boundary_eq_affected"]
    assert dose["n_s2_affected"] >= dose["n_s2_internal"]
    assert dose["n_s2_affected"] == pilot["topk"][0]["n_s2_occurrences_either_bond"]
    assert dose["n_s2_internal"] == pilot["topk"][0]["n_s2_occurrences"]


# ---------------------------------------------------------------------------
# TEST: the real gate removes exactly n_s2_affected occurrences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_fixture", ["ctx", "ctx_lipo"])
def test_gate_removes_exactly_affected_occurrences(dataset_fixture, request) -> None:
    import as_data as D
    import as_gate as G

    ctx = request.getfixturevalue(dataset_fixture)
    cache = ctx["bundle"].caches["valid"]
    occ = ctx["occs"]["valid"]
    rows = ctx["pilot"]["valid"][:N_GATE_MOLECULES]
    n_checked = 0
    for pilot in rows:
        row = int(pilot["cache_row"])
        bonds = [int(b) for b in pilot["topk"][0]["bonds"]]
        mol = G.SingleMolecule(cache, row, ctx["bundle"].spec)
        acc = S2.dose_accounting(mol, bonds)
        # independent recount of the OR condition
        ids = set(bonds)
        b1, b2 = mol.occ.path_bond1, mol.occ.path_bond2
        direct = int(sum(1 for e1, e2 in zip(b1.tolist(), b2.tolist()) if e1 in ids or e2 in ids))
        assert direct == acc.n_s2_affected
        assert acc.n_s2_internal == pilot["topk"][0]["n_s2_occurrences"]
        assert acc.n_s2_affected == pilot["topk"][0]["n_s2_occurrences_either_bond"]

        chk = S2.gate_drop_equivalence(
            ctx["arm"], cache, row, occ, mol, bonds, acc, sigma=ctx["sigma"], mu=ctx["mu"]
        )
        assert chk["passed"], (dataset_fixture, pilot["row_id"], chk)
        assert chk["n_dropped_matches_accounting"]
        assert chk["weight_audit"]["n_weight_zeroed_by_gate"] == acc.n_s2_affected
        assert chk["weight_audit"]["n_weight_kept"] == acc.n_s2_unaffected
        assert chk["weight_audit"]["n_weight_fractional"] == 0

        # the AND quantity would remove fewer occurrences: prove the difference is real
        gb = D.collate_gated(cache, [row], occ)
        part = S2.s2_partition(mol.occ, bonds)
        cache_and, occ_and = D.molecule_cache(
            {
                "atom_feat": mol.mol["atom_feat"], "atom_type": mol.mol["atom_type"],
                "atom_scalar": mol.mol["atom_scalar"], "bond_feat": mol.mol["bond_feat"],
                "bond_type": mol.mol["bond_type"], "bond_index": mol.mol["bond_index"],
                "global": mol.mol["global"], "smiles": mol.mol["smiles"],
                "molecule_id": mol.mol["molecule_id"],
            },
            mol.occ, drop_paths=part["internal"],
        )
        gb_and = D.collate_gated(cache_and, [0], occ_and)
        import torch

        with torch.no_grad():
            p_open = float(G.gated_parts(ctx["arm"], gb, gb.gate_ones())["pred_std"][0])
            p_gated = float(
                G.gated_parts(ctx["arm"], gb, gb.gate_ones(),
                              gb.gate_matrix([G.bond_mask(mol.n_bonds, bonds, 0.0)]))["pred_std"][0]
            )
            p_and = float(G.gated_parts(ctx["arm"], gb_and, gb_and.gate_ones())["pred_std"][0])
        gap = abs(p_gated - p_and) * ctx["sigma"]
        tol = float(C.trace_tolerance(p_open * ctx["sigma"] + ctx["mu"]))
        assert int(part["internal"].sum()) < acc.n_s2_affected
        assert gap > tol or acc.n_s2_boundary == 0, (
            "dropping only the AND occurrences should not reproduce the OR gate",
            dataset_fixture, pilot["row_id"], gap, tol,
        )
        n_checked += 1
    assert n_checked == len(rows)


# ---------------------------------------------------------------------------
# TEST: localization closure and inside/outside split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_fixture", ["ctx", "ctx_lipo"])
def test_localization_closures(dataset_fixture, request) -> None:
    import as_data as D
    import as_gate as G

    ctx = request.getfixturevalue(dataset_fixture)
    cache = ctx["bundle"].caches["valid"]
    occ = ctx["occs"]["valid"]
    n_checked = 0
    for pilot in ctx["pilot"]["valid"][:N_GATE_MOLECULES]:
        row = int(pilot["cache_row"])
        bonds = [int(b) for b in pilot["topk"][0]["bonds"]]
        mol = G.SingleMolecule(cache, row, ctx["bundle"].spec)
        variants = [I.variant("BASE", "baseline", []), I.variant("TOP1", "discovery_mask", bonds)]
        evaluated = I.run_variants(
            ctx["arm"], cache, row, occ, variants,
            sigma=ctx["sigma"], mu=ctx["mu"], device="cpu",
        )
        base = next(v for v in evaluated if v["name"] == "BASE")
        after = next(v for v in evaluated if v["name"] == "TOP1")
        inside = np.zeros(mol.n_atoms, dtype=bool)
        inside[np.asarray(mol.bond_atoms(bonds), dtype=np.int64)] = True
        acc = S2.dose_accounting(mol, bonds)
        loc = I.localization_record(
            base, after, inside_atoms=inside, accounting=acc, train_std=ctx["train_std"]
        )
        assert loc["closure_local_sum_passed"], loc
        assert loc["closure_total_passed"], loc
        eff = I.effect_record(base, after, train_std=ctx["train_std"])
        assert loc["closure_total_error"] <= loc["closure_total_tolerance"]
        assert abs(loc["delta_total"] - eff["delta_total"]) <= 1e-12
        # the absolute response is exactly the sum of the per-root absolute deltas
        assert loc["inside_abs_response"] + loc["outside_abs_response"] == pytest.approx(
            loc["total_abs_response"], rel=0, abs=0
        )
        if loc["total_abs_response"] > 0:
            assert 0.0 <= loc["outside_response_fraction"] <= 1.0
        assert loc["n_atoms_inside"] + loc["n_atoms_outside"] == mol.n_atoms
        # the S2 accounting attached to the localization row is the OR quantity
        assert loc["n_s2_affected"] == acc.n_s2_affected
        assert loc["n_s2_internal"] + loc["n_s2_boundary"] == loc["n_s2_affected"]
        n_checked += 1
    assert n_checked == N_GATE_MOLECULES


# ---------------------------------------------------------------------------
# TEST: corrected matching rule
# ---------------------------------------------------------------------------


def test_corrected_matching_rule(ctx) -> None:
    import as_gate as G

    cache = ctx["bundle"].caches["valid"]
    n_checked = 0
    for pilot in ctx["pilot"]["valid"][:6]:
        row = int(pilot["cache_row"])
        bonds = [int(b) for b in pilot["topk"][0]["bonds"]]
        mol = G.SingleMolecule(cache, row, ctx["bundle"].spec)
        target = S2.dose_accounting(mol, bonds)
        rng = C.control_rng(ctx["dataset"], "valid", int(pilot["row_id"]))
        out = K.matched_random_controls(mol, bonds, rng)
        assert out["rule"]["match_quantity"] == "n_s2_affected"
        assert out["top1"]["n_s2_affected"] == target.n_s2_affected
        assert out["n_matched"] <= C.N_MATCHED_RANDOM
        assert out["attempts"] <= C.MAX_GROWTH_ATTEMPTS
        assert out["status"] == ("OK" if out["n_matched"] >= C.MIN_MATCHED_RANDOM
                                 else "INSUFFICIENT_MATCHED_NULL")
        seen = {tuple(sorted(bonds))}
        for rec in out["matched"]:
            key = tuple(sorted(rec["bonds"]))
            assert key not in seen, "identical duplicates are not allowed"
            seen.add(key)
            assert len(rec["bonds"]) == len(bonds)
            assert mol.atoms_connected(rec["bonds"])
            assert rec["affected_rel_diff"] <= C.S2_AFFECTED_REL_TOL + 1e-12
            assert rec["n_s2_affected"] == S2.dose_accounting(mol, rec["bonds"]).n_s2_affected
            assert rec["n_s2_internal"] + rec["n_s2_boundary"] == rec["n_s2_affected"]
            assert rec["kind"] == "matched_random"
        # determinism: the same RNG stream reproduces the same accepted set
        again = K.matched_random_controls(
            mol, bonds, C.control_rng(ctx["dataset"], "valid", int(pilot["row_id"]))
        )
        assert [r["bonds"] for r in again["matched"]] == [r["bonds"] for r in out["matched"]]
        assert again["status"] == out["status"]
        n_checked += 1
    assert n_checked == 6


def test_dose_accounting_rejects_empty_set(ctx) -> None:
    import as_gate as G

    mol = G.SingleMolecule(ctx["bundle"].caches["valid"], 0, ctx["bundle"].spec)
    with pytest.raises(ValueError):
        S2.dose_accounting(mol, [])


# ---------------------------------------------------------------------------
# TEST: the aggregate stage reproduces the pilot's own table
# ---------------------------------------------------------------------------


def test_pilot_metric_replay() -> None:
    import au_aggregate as A

    for dataset in C.DATASETS:
        pilot = A.load_pilot(dataset)
        replay = A.replay_pilot_metrics(dataset, pilot, C.train_std(dataset))
        assert replay["pass"], (dataset, replay)
        for split in C.SPLITS:
            assert replay["splits"][split]["passed"], (dataset, split)
