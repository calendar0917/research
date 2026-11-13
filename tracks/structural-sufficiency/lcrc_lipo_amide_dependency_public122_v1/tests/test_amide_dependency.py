"""Small numerical / structural fixtures for the amide-dependency protocol.

These tests are deliberately tiny and self-contained: no training, no GPU, no
official-test protocol run.  Tests that need the frozen data / checkpoint /
produced artifacts skip when those bytes are absent (fresh-clone safe).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import common as C  # noqa: E402
import motif as M  # noqa: E402
import prepare as P  # noqa: E402
import views as V  # noqa: E402


# ---------------------------------------------------------------------------
# structure fixtures (toy raw rows, no dataset)
# ---------------------------------------------------------------------------


def _toy_row(smiles: str, row_id: int = 0) -> dict:
    pytest.importorskip("rdkit")
    return C.RD.smiles_to_raw_row(smiles, row_id)


def test_motif_on_n_methylacetamide() -> None:
    row = _toy_row("CC(=O)NC")
    g = M.summarise_raw(row, 0)
    inst = M.find_instances(g)
    assert len(inst) == 1
    i = inst[0]
    # RDKit atom order of "CC(=O)NC": C0 C1 O2 N3 C4
    assert (i.c, i.o, i.n, i.acyl_c) == (1, 2, 3, 0)
    assert (i.degree_n, i.degree_acyl, i.ring_n, i.ring_acyl) == (2, 1, False, False)
    # MATCHED fails (degree mismatch), BALANCED holds (2 vs 1 heavy atoms)
    assert i.matched is False
    assert (i.n_side, i.acyl_side) == (2, 1)
    assert abs(i.n_side - i.acyl_side) <= C.MATCHED_SIDE_TOL


def test_motif_allows_an_n_inside_a_ring() -> None:
    # Amide N inside a piperidine ring: the c-N link is exocyclic and therefore a
    # bridge, so the instance is allowed; the N itself *is* a ring atom.
    row = _toy_row("CC(=O)N1CCCCC1")
    g = M.summarise_raw(row, 0)
    inst = M.find_instances(g)
    assert len(inst) == 1
    assert inst[0].ring_n is True
    assert g.is_bridge[inst[0].bond_cn] is np.True_ or bool(g.is_bridge[inst[0].bond_cn])
    # A formyl C carries one H (raw atom_h = 1), so it is not a target instance.
    g2 = M.summarise_raw(_toy_row("C1CCN(C1)C=O"), 0)
    assert M.find_instances(g2) == []


def test_ring_atom_definition_matches_cycle_membership() -> None:
    row = _toy_row("c1ccccc1CC(=O)N")
    g = M.summarise_raw(row, 0)
    aromatic = [i for i in range(g.n_atoms) if int(g.atom_class[i]) == 3]
    assert all(bool(g.ring_atom[i]) for i in aromatic)
    n_atom = [i for i in range(g.n_atoms) if int(g.atom_class[i]) == 7][0]
    assert not bool(g.ring_atom[n_atom])


def test_cut_removes_exactly_one_physical_bond() -> None:
    row = _toy_row("CC(=O)NCCc1ccccc1")
    pytest.importorskip("feature_spec")
    from dissect_data import frozen_spec

    g = M.summarise_raw(row, 0)
    inst = M.find_instances(g)[0]
    cut, info = V.cut_row(row, inst.c, inst.n)
    assert info["removed_directed_records"] == 2
    assert np.array_equal(np.asarray(cut["atom_class"]), np.asarray(row["atom_class"]))
    chk = V.verify_cut(row, cut, cut_a=inst.c, cut_b=inst.n, spec=frozen_spec())
    assert chk["removed_is_requested_cut"]
    assert not chk["added_physical_bonds"] and not chk["changed_bond_attributes"]
    assert all(chk["atom_attributes_unchanged"].values())
    assert chk["components_increased_by_one"]
    summ = V.cut_effect_summary(chk, kind="T", c=inst.c, n=inst.n, o=inst.o, acyl_c=inst.acyl_c)
    assert summ["pass"], summ


def test_k_cut_keeps_the_carbonyl_nitrogen_path() -> None:
    from dissect_data import frozen_spec

    row = _toy_row("CC(=O)NCC")
    g = M.summarise_raw(row, 0)
    inst = M.find_instances(g)[0]
    cut, _ = V.cut_row(row, inst.c, inst.acyl_c)
    chk = V.verify_cut(row, cut, cut_a=inst.c, cut_b=inst.acyl_c, spec=frozen_spec())
    summ = V.cut_effect_summary(chk, kind="K", c=inst.c, n=inst.n, o=inst.o, acyl_c=inst.acyl_c)
    assert summ["pass"], summ
    assert summ["dist_c_n_after"] == 1


def test_atom_permutation_preserves_instances() -> None:
    row = _toy_row("CC(=O)Nc1ccc(CC(=O)NC)cc1")
    g = M.summarise_raw(row, 0)
    n = g.n_atoms
    perm = np.random.default_rng(7).permutation(n)
    prow = V.permute_row(row, perm)
    found = {(i.c, i.o, i.n, i.acyl_c) for i in M.find_instances(M.summarise_raw(prow, 0))}
    expected = {
        (int(perm[i.c]), int(perm[i.o]), int(perm[i.n]), int(perm[i.acyl_c])) for i in M.find_instances(g)
    }
    assert found == expected
    # and the raw attributes are permuted consistently
    assert np.array_equal(np.asarray(prow["atom_class"]), np.asarray(row["atom_class"])[np.argsort(perm)])


def test_bridges_match_networkx() -> None:
    nx = pytest.importorskip("networkx")
    row = _toy_row("c1ccc2c(c1)CC(=O)N2C")
    g = M.summarise_raw(row, 0)
    graph = nx.Graph()
    graph.add_nodes_from(range(g.n_atoms))
    graph.add_edges_from(zip(g.bu.tolist(), g.bv.tolist()))
    expected = {tuple(sorted((int(a), int(b)))) for a, b in nx.bridges(graph)}
    got = {(int(u), int(v)) for u, v, flag in zip(g.bu.tolist(), g.bv.tolist(), g.is_bridge.tolist()) if flag}
    assert got == expected


# ---------------------------------------------------------------------------
# loss / weighting / bootstrap fixtures
# ---------------------------------------------------------------------------


def _panel(l0, lt, lk, n_inst, lr=None, sq=None):
    l0 = np.asarray(l0, dtype=float)
    lt = np.asarray(lt, dtype=float)
    lk = np.asarray(lk, dtype=float)
    lr = np.full(l0.shape, np.nan) if lr is None else np.asarray(lr, dtype=float)
    n_inst = np.asarray(n_inst, dtype=np.int64)
    return {
        "rows": np.arange(l0.size, dtype=np.int64),
        "n_instances": n_inst,
        "n_r_candidates": np.where(np.isfinite(lr), n_inst, 0).astype(np.int64),
        "L0": l0,
        "LT": lt,
        "LK": lk,
        "LR": lr,
        "sq0": l0**2,
        "sqT": lt**2,
        "sqK": lk**2,
        "sqR": lr**2,
        "move_abs_T": np.zeros_like(l0),
        "move_signed_T": np.zeros_like(l0),
        "move_abs_K": np.zeros_like(l0),
        "move_signed_K": np.zeros_like(l0),
        "move_abs_R": np.zeros_like(l0),
        "move_signed_R": np.zeros_like(l0),
        "L0_Rsub": np.where(np.isfinite(lr), l0, np.nan),
        "LT_Rsub": np.where(np.isfinite(lr), lt, np.nan),
        "B": lt - l0,
        "D": lt - lk,
        "B_R": lr - l0,
        "D_R": lt - lr,
    }


def test_B_D_sign_and_magnitude() -> None:
    import analysis

    # molecule 0: target cut is worse than the control cut and than O
    panel = _panel([0.2, 0.1], [0.6, 0.3], [0.1, 0.3], [2, 1])
    pe = analysis.point_estimates(panel)
    assert pe["B"] == pytest.approx(0.3)
    assert pe["D"] == pytest.approx(0.25)
    assert pe["MAE_O"] == pytest.approx(0.15)
    assert pe["MAE_T"] == pytest.approx(0.45)
    assert pe["n_instances"] == 3
    # sign flips when the control cut is the damaging one
    flipped = _panel([0.2], [0.1], [0.9], [1])
    assert analysis.point_estimates(flipped)["D"] == pytest.approx(-0.8)


def test_molecule_and_instance_weighting() -> None:
    import analysis

    # two molecules: 1 instance and 3 instances.  Molecules are equally
    # weighted, so B = (0.5 + 0.0) / 2 and not (0.5*1 + 0.0*3)/4.
    panel = _panel([0.0, 0.0], [0.5, 0.0], [0.5, 0.0], [1, 3])
    pe = analysis.point_estimates(panel)
    assert pe["B"] == pytest.approx(0.25)


def test_tied_remote_candidates_average_at_the_loss_layer() -> None:
    import analysis

    # one molecule, one instance, two tied candidates with predictions 0 and 4
    # for y = 1: the instance loss is mean(|0-1|, |4-1|) = 2, not |2-1| = 1.
    inst = [
        {
            "row_id": 0,
            "c": 1,
            "view_t": "0:T",
            "view_k": "0:K",
            "views_r": ["0:R:a", "0:R:b"],
            "matched": 1,
            "balanced": 1,
            "views_r_ok": True,
        },
    ]
    preds = {
        ("0:O", "FULL"): 1.0,
        ("0:T", "FULL"): 1.0,
        ("0:K", "FULL"): 1.0,
        ("0:R:a", "FULL"): 0.0,
        ("0:R:b", "FULL"): 4.0,
    }
    y = {0: 1.0}
    d = analysis.panel("MATCHED", "test", "FULL", inst=inst, y=y, preds=preds, split_ids={0})
    assert d["LR"][0] == pytest.approx(2.0)
    assert d["LT"][0] == pytest.approx(0.0)
    assert d["B_R"][0] == pytest.approx(2.0)


def test_bootstrap_resamples_whole_clusters() -> None:
    import analysis

    rows = np.asarray([10, 11, 12], dtype=np.int64)
    scaffold = {10: "A", 11: "A", 12: "B"}
    cw = analysis.cluster_weights(rows, scaffold, rng=np.random.default_rng(0), replicates=64)
    W = cw["weights"]
    assert cw["n_clusters"] == 2
    assert W.shape == (64, 3)
    # molecules 10 and 11 share a cluster and therefore the same multiplicity
    assert np.array_equal(W[:, 0], W[:, 1])
    # each replicate draws exactly K clusters (with replacement)
    assert np.all(cw["cluster_draw_counts"].sum(axis=1) == 2)
    assert np.array_equal(cw["cluster_draw_counts"][:, 0], W[:, 0])
    assert cw["max_cluster_share"] == pytest.approx(2 / 3)


def test_weighted_stats_use_the_same_resample_for_all_views() -> None:
    import analysis

    panel = _panel([0.1, 0.2], [0.3, 0.2], [0.2, 0.2], [1, 1])
    cw = analysis.cluster_weights(
        panel["rows"], {0: "A", 1: "B"}, rng=np.random.default_rng(1), replicates=32
    )
    reps = analysis.weighted_stats(cw["weights"], panel)
    # with all draws on one cluster the replicate mean equals that molecule's value
    single = np.zeros((1, 2))
    single[0, 0] = 1.0
    one = analysis.weighted_stats(single, panel)
    assert one["B"][0] == pytest.approx(0.2)
    assert one["D"][0] == pytest.approx(0.1)
    assert np.isfinite(reps["B"]).all()


def test_target_norm_roundtrip() -> None:
    mean, scale = 2.176985119047619, 1.2091154725538142
    y = np.asarray([-1.18, 0.0, 3.69])
    z = (y - mean) / scale
    assert np.allclose(mean + scale * z, y)


def test_pair_direction_rule() -> None:
    import pairs

    assert pairs.orient_pair(5, 7, 1, 2) == (7, 5, 2, 1)
    assert pairs.orient_pair(5, 7, 2, 1) == (5, 7, 2, 1)


# ---------------------------------------------------------------------------
# artifact-level regression guards (skip when the frozen bytes are absent)
# ---------------------------------------------------------------------------


def test_reference_metrics_match_the_frozen_predictions() -> None:
    if not C.REFERENCE_PREDICTIONS.exists():
        pytest.skip("frozen predictions npz missing")
    d = np.load(C.REFERENCE_PREDICTIONS, allow_pickle=True)
    valid = float(np.sqrt(np.mean((d["valid_pred_raw"] - d["valid_y"]) ** 2)))
    test = float(np.sqrt(np.mean((d["test_pred_raw"] - d["test_y"]) ** 2)))
    assert valid == pytest.approx(C.REFERENCE["valid_rmse"], abs=1e-12)
    assert test == pytest.approx(C.REFERENCE["test_rmse"], abs=1e-12)
    assert float(np.mean(np.abs(d["test_pred_raw"] - d["test_y"]))) == pytest.approx(
        C.REFERENCE["test_mae"], abs=1e-12
    )


def test_checkpoint_identity() -> None:
    if not C.CHECKPOINT.exists():
        pytest.skip("frozen checkpoint missing")
    ck = C.load_checkpoint()
    assert C.DM.state_fingerprint(ck["state_dict"]) == C.STATE_HASH
    assert int(ck["parameters"]) == C.EXPECTED_PARAMS


def test_structural_precounts_are_frozen() -> None:
    path = C.ANALYSIS / "motif_catalog.csv"
    if not path.exists():
        pytest.skip("motif catalog missing")
    rows = P.read_csv(path)
    counts = {"ALL": {"molecules": {}, "instances": {}}}
    for split in C.SPLITS:
        sel = [r for r in rows if r["split"] == split]
        counts["ALL"]["molecules"][split] = len({r["row_id"] for r in sel})
        counts["ALL"]["instances"][split] = len(sel)
    for split in C.SPLITS:
        assert counts["ALL"]["molecules"][split] == P.EXPECTED_COUNTS["ALL"]["molecules"][split]
        assert counts["ALL"]["instances"][split] == P.EXPECTED_COUNTS["ALL"]["instances"][split]


def test_results_are_self_consistent() -> None:
    """B/D in results.csv must equal the difference of the view MAEs."""
    path = C.ANALYSIS / "results.csv"
    if not path.exists():
        pytest.skip("results.csv missing")
    rows = P.read_csv(path)
    table: dict[tuple, dict[str, float]] = {}
    for r in rows:
        table.setdefault((r["population"], r["split"], r["scope"]), {})[r["metric"]] = float(r["value"])
    checked = 0
    for key, d in table.items():
        if not {"MAE_O", "MAE_T", "MAE_K", "B", "D"} <= set(d):
            continue
        assert d["B"] == pytest.approx(d["MAE_T"] - d["MAE_O"], abs=1e-9)
        assert d["D"] == pytest.approx(d["MAE_T"] - d["MAE_K"], abs=1e-9)
        checked += 1
    assert checked >= 8
