"""Minimal correctness gates for the frozen benchmark-positioning protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import bp_common as C  # noqa: E402
import bp_data as D  # noqa: E402
import bp_graph as BG  # noqa: E402
import bp_metrics as MET  # noqa: E402
import bp_model as BM  # noqa: E402
import bp_spec as S  # noqa: E402
import bp_tu as T  # noqa: E402
import lcrc_data as PREP_DATA  # noqa: E402
import molecular_graph as MG  # noqa: E402
import tu_model as M  # noqa: E402


@pytest.fixture(scope="session")
def mutag_scan():
    return T.scan_dataset("MUTAG")


@pytest.fixture(scope="session")
def mutag_spec(mutag_scan):
    node_types = T._node_type_table(mutag_scan)
    bond_types = T._bond_type_table(mutag_scan)
    return (
        S.tu_spec(
            "MUTAG-test",
            node_types["names"],
            int(node_types["unk_index"]),
            bond_types["names"],
            int(bond_types["unk_id"]),
            continuous_atom_dim=0,
        ),
        node_types,
        bond_types,
    )


@pytest.fixture(scope="session")
def tiny_cache(mutag_scan, mutag_spec):
    spec, node_types, bond_types = mutag_spec
    mols = []
    for rec in mutag_scan["records"][:12]:
        mols.append(T._build_molecule(rec, spec, node_types, bond_types, "MUTAG")["mol"])
    cache = PREP_DATA.concat_molecules(mols, spec, split="unit-test")
    return cache, spec


def test_tu_splits_disjoint_cover_and_seeds(mutag_scan):
    y = np.asarray(mutag_scan["y"], dtype=np.int64)
    folds = T.build_folds(y, "MUTAG")
    assert folds["n_splits"] == 10
    assert folds["random_state"] == C.TU_OUTER_SPLIT_SEED
    all_rows = np.arange(len(y), dtype=np.int64)
    for fold in folds["folds"]:
        tr = np.asarray(fold["rows"]["train"], dtype=np.int64)
        va = np.asarray(fold["rows"]["valid"], dtype=np.int64)
        te = np.asarray(fold["rows"]["test"], dtype=np.int64)
        assert not (set(tr.tolist()) & set(va.tolist()))
        assert not (set(tr.tolist()) & set(te.tolist()))
        assert not (set(va.tolist()) & set(te.tolist()))
        assert np.array_equal(np.sort(np.concatenate([tr, va, te])), all_rows)
        assert fold["model_seed"] == C.TU_MODEL_SEED_BASE + fold["fold"]
        assert fold["inner_valid_seed"] == C.TU_INNER_VALID_SEED_BASE + fold["fold"]
        assert len(tr) + len(va) + len(te) == len(y)


def test_tu_adapters_keep_attributes_without_dropping():
    mutag = T.scan_dataset("MUTAG")
    assert mutag["stats"]["has_edge_labels"] is True
    assert mutag["stats"]["has_continuous_node_attributes"] is False
    proteins = T.scan_dataset("PROTEINS")
    assert proteins["stats"]["has_continuous_node_attributes"] is True
    assert proteins["stats"]["continuous_node_attribute_dim"] == 1
    enzymes = T.scan_dataset("ENZYMES")
    assert enzymes["stats"]["continuous_node_attribute_dim"] == 18
    for name in ("PROTEINS", "ENZYMES", "MUTAG"):
        assert T.scan_dataset(name)["stats"]["has_continuous_edge_attributes"] is False


def _small_triangle_tail():
    pairs = np.asarray([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=np.int64)
    graph = MG.build_molecular_graph(
        atom_type=np.asarray([0, 1, 2, 3], dtype=np.int64),
        bond_index=pairs,
        bond_type=np.asarray([1, 1, 1, 2], dtype=np.int64),
        atom_feat=np.zeros((4, 1), dtype=np.float32),
        bond_feat=np.zeros((4, 1), dtype=np.float32),
        molecule_id="triangle-tail",
    )
    return graph


def test_s2_matches_hand_count_formula_and_upstream():
    graph = _small_triangle_tail()
    analysis = MG.analyse_graph(graph)
    s2 = BG.s2_arrays(graph.atom_type, analysis.adj, 2)
    per_root = np.diff(s2["path_ptr"])
    assert per_root.tolist() == [3, 3, 2, 2]
    assert int(s2["path_ptr"][-1]) == 10
    formula = BG.s2_counts_from_formula(analysis.adj)
    assert formula.tolist() == per_root.tolist()
    upstream = MG.enumerate_s2(graph.atom_type, analysis.adj, 2, analysis.dist)
    up_flat = [rec for row in upstream for rec in row]
    assert len(up_flat) == 10
    ours = sorted(
        (int(c), int(e), int(cl))
        for c, e, cl in zip(s2["path_cat"], s2["path_end63"], s2["path_closure"])
    )
    theirs = sorted((int(r["path_cat"]), int(r["end63"]), int(r["closure"])) for r in up_flat)
    assert ours == theirs
    fast = BG.analyse_graph_fast(graph)
    assert BG.compare_with_upstream(graph, fast, analysis, 2)["s2_counts_equal"] is True


def test_fast_analysis_matches_upstream_on_small_graphs(mutag_scan):
    for rec in mutag_scan["records"][:4]:
        n = rec["n_atoms"]
        graph = MG.build_molecular_graph(
            atom_type=np.zeros(n, dtype=np.int64),
            bond_index=rec["pairs"],
            bond_type=np.ones(len(rec["pairs"]), dtype=np.int64),
            atom_feat=np.zeros((n, 1), dtype=np.float32),
            bond_feat=np.zeros((len(rec["pairs"]), 1), dtype=np.float32),
        )
        up = MG.analyse_graph(graph)
        fast = BG.analyse_graph_fast(graph)
        assert np.array_equal(up.min_ring_size, fast.min_ring_size)
        n_d2 = np.asarray([np.sum(up.dist[s] == 2) for s in range(n)], dtype=np.int64)
        assert np.array_equal(n_d2, fast.n_d2)
        assert up.n_components == fast.n_components
        assert up.n_ring_systems == fast.n_ring_systems


def test_cache_has_no_target_labels(tiny_cache):
    cache, _spec = tiny_cache
    forbidden = {"y", "label", "labels", "target", "targets", "smiles_target"}
    assert not (forbidden & set(cache.keys()))
    assert cache["meta"]["labels_present"] is False


def test_train_only_normalisation_and_continuous_norm():
    atom_ptr = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
    bond_ptr = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    atom_scalar = np.asarray(
        [[0, 10], [0, 10], [5, 20], [5, 20], [100, 30], [100, 30], [200, 40], [200, 40]],
        dtype=np.float32,
    )
    global_ = np.asarray([[0, 1], [5, 2], [100, 3], [200, 4]], dtype=np.float32)
    atom_feat = np.asarray(
        [[1, 10], [3, 14], [5, 18], [7, 22], [100, 30], [200, 40], [300, 50], [400, 60]],
        dtype=np.float32,
    )
    cache = {
        "n_molecules": 4,
        "atom_ptr": atom_ptr,
        "bond_ptr": bond_ptr,
        "atom_scalar": atom_scalar,
        "global": global_,
        "atom_feat": atom_feat,
    }
    class Spec:  # minimal duck type for the norm functions
        n_scalars = 2
        global_dim = 2
    spec = Spec()
    train_rows = np.asarray([0, 1], dtype=np.int64)
    norm = D.fit_norm_rows(cache, spec, train_rows, source="unit-test")
    assert np.allclose(norm["scalar_mean"], atom_scalar[:4].mean(0))
    assert np.allclose(norm["global_mean"], global_[train_rows].mean(0))
    # Changing non-train rows must not change the fitted values.
    other = dict(cache)
    other["atom_scalar"] = atom_scalar.copy()
    other["atom_scalar"][4:] = 999.0
    other["global"] = global_.copy()
    other["global"][2:] = 999.0
    norm2 = D.fit_norm_rows(other, spec, train_rows, source="unit-test")
    assert np.allclose(norm["scalar_mean"], norm2["scalar_mean"])
    assert np.allclose(norm["global_mean"], norm2["global_mean"])
    cont = D.fit_continuous_atom_norm(cache, train_rows, start=0, dim=2)
    assert np.allclose(cont["mean"], atom_feat[:4].mean(0))
    assert not np.allclose(cont["mean"], atom_feat.mean(0))


def test_single_graph_and_batch_forward_are_equal(tiny_cache):
    cache, spec = tiny_cache
    norm = {
        "global_mean": np.zeros(spec.global_dim, np.float32),
        "global_scale": np.ones(spec.global_dim, np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, np.float32),
        "scalar_scale": np.ones(spec.n_scalars, np.float32),
    }
    model, info = BM.build_c1(spec, norm, 2, 123, "cpu")
    assert info["shared_with_full"]["all_equal"] is True
    model.eval()
    one = D.collate(cache, [0])
    batch = D.collate(cache, [0, 1, 2])
    with torch.no_grad():
        a = model(one).numpy()
        b = model(batch).numpy()
    assert np.allclose(a[0], b[0], atol=1e-6)


def test_c1_is_full_minus_b_mean(tiny_cache):
    _cache, spec = tiny_cache
    full = M.local_layout(spec, "FULL")
    c1 = M.local_layout(spec, "C1")
    assert len(M.local_columns(spec, "FULL")) - len(M.local_columns(spec, "C1")) == 96
    assert "b_mean" not in c1
    assert "b_mean" in full
    assert set(full) - set(c1) == {"b_mean"}
    norm = {
        "global_mean": np.zeros(spec.global_dim, np.float32),
        "global_scale": np.ones(spec.global_dim, np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, np.float32),
        "scalar_scale": np.ones(spec.n_scalars, np.float32),
    }
    model, info = BM.build_c1(spec, norm, 2, 7, "cpu")
    assert info["variant"] == "C1"
    assert "local.b_mean" not in dict(model.named_parameters())
    assert model.local.variant == "C1"


def test_checkpoint_reload_reproduces_logits(tiny_cache):
    cache, spec = tiny_cache
    norm = {
        "global_mean": np.zeros(spec.global_dim, np.float32),
        "global_scale": np.ones(spec.global_dim, np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, np.float32),
        "scalar_scale": np.ones(spec.n_scalars, np.float32),
    }
    model, _ = BM.build_c1(spec, norm, 2, 11, "cpu")
    model.eval()
    batch = D.collate(cache, [0, 1])
    with torch.no_grad():
        before = model(batch).numpy()
    path = Path("/tmp/bp_test_checkpoint.pt")
    torch.save(model.state_dict(), path)
    model2, _ = BM.build_c1(spec, norm, 2, 11, "cpu")
    model2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model2.eval()
    with torch.no_grad():
        after = model2(batch).numpy()
    assert np.allclose(before, after, atol=1e-7)


def test_metrics_recompute_and_missing_values():
    logits = np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    y = np.asarray([0, 2, 1, 2], dtype=np.int64)
    m = MET.classification_metrics(logits, y, 3)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["macro_f1"] >= 0.0
    assert np.asarray(m["confusion_matrix"]).shape == (3, 3)
    mask = np.asarray([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64)
    pred = np.zeros((2, 2), dtype=np.float32)
    loss = MET.masked_bce_loss(torch.as_tensor(pred), torch.as_tensor(mask))
    assert torch.isfinite(loss).item()
    y_auc = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float64).reshape(-1, 1)
    p_auc = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6], dtype=np.float64).reshape(-1, 1)
    auc = MET.ogb_rocauc("ogbg-molhiv", y_auc, p_auc)
    assert 0.0 <= float(auc["value"]) <= 1.0
    ap = MET.macro_ap(
        np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float64),
        np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.2]], dtype=np.float64),
    )
    assert np.isfinite(ap["value"])


def test_peptide_one_hot_categories_and_types():
    atom_vocabs = S.PEPTIDE_ATOM_VOCABS
    bond_vocabs = S.PEPTIDE_BOND_VOCABS
    atom_raw = np.zeros((4, len(atom_vocabs)), dtype=np.int64)
    bond_raw = np.zeros((3, len(bond_vocabs)), dtype=np.int64)
    atom_raw[:, 0] = [0, 1, 2, 3]
    bond_raw[:, 0] = [0, 1, 2]
    graph = MG.build_molecular_graph(
        atom_type=atom_raw[:, 0],
        bond_index=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        bond_type=bond_raw[:, 0] + 1,
        atom_feat=np.zeros((4, 1), dtype=np.float32),
        bond_feat=np.zeros((3, 1), dtype=np.float32),
    )
    analysis = BG.analyse_graph_fast(graph)
    atom_feat, bond_feat = S.peptide_features(
        atom_raw, bond_raw, atom_raw[:, 0], bond_raw[:, 0] + 1, analysis
    )
    assert atom_feat.shape[0] == 4 and bond_feat.shape[0] == 3
    offset = 0
    for width in atom_vocabs:
        assert np.all(atom_feat[:, offset : offset + width].sum(1) == 1)
        offset += width
    offset = 0
    for width in bond_vocabs:
        assert np.all(bond_feat[:, offset : offset + width].sum(1) == 1)
        offset += width
    bad = atom_raw.copy()
    bad[0, 0] = atom_vocabs[0]
    with pytest.raises(Exception):
        S.peptide_features(bad, bond_raw, bad[:, 0], bond_raw[:, 0] + 1, analysis)
