"""Native export / replay and tree-path accounting."""

from __future__ import annotations

import numpy as np

import sq_ast as A
import sq_bank as B
import sq_common as C
import sq_explain as E
import sq_xgb as X
from helpers import make_packed, random_tree_molecules


def _toy_model():
    rng = np.random.default_rng(0)
    graphs = random_tree_molecules(rng, 40)
    packed = make_packed(graphs)
    y = (
        2.0 * packed.graph_counts(A.evaluate(packed, A.ATOM_EQ(0, 7)))
        + 0.5 * packed.graph_counts(A.evaluate(packed, A.DEG_EQ(1)))
        + rng.normal(scale=0.05, size=packed.n_graphs)
    ).astype(np.float64)
    scaler = {"mean": float(y.mean()), "std": float(y.std(ddof=0))}
    records = [
        B.QueryRecord(ast=A.ATOM_EQ(0, 7), label="Q000", source="B", outer_round=0, train_support=0),
        B.QueryRecord(ast=A.DEG_EQ(1), label="Q001", source="B", outer_round=0, train_support=0),
    ]
    design, names = B.design_matrix(packed, records, include_base=True)
    head = X.run_head(
        "A",
        "H1",
        {"train": design, "valid": design, "test": design},
        {"train": y, "valid": y, "test": y},
        scaler,
        names,
        device="cpu",
        num_boost_round=32,
    )
    return packed, design, names, scaler, y, head


def test_native_export_replays_xgboost_exactly():
    packed, design, names, scaler, y, head = _toy_model()
    rounds = head.best["iteration"]
    trees, audit = E.export_and_audit(
        booster=head.booster,
        feature_names=names,
        matrices={"train": design, "valid": design, "test": design},
        scaler=scaler,
        row_ids={"train": np.arange(y.size), "valid": np.arange(y.size), "test": np.arange(y.size)},
        rounds=rounds,
    )
    assert trees["n_trees"] == rounds
    assert audit["pass"], audit
    assert audit["max_abs_error_overall"] <= 1e-5
    # the exported model must also obey the feature-name contract
    replay = X.replay_native(trees, design, names, scaler)
    expected = head.booster.predict(
        E.dmatrix(design, names), iteration_range=(0, rounds)
    ) * scaler["std"] + scaler["mean"]
    assert np.max(np.abs(replay - expected)) <= 1e-5


def test_tree_paths_sum_to_the_prediction():
    packed, design, names, scaler, y, head = _toy_model()
    rounds = head.best["iteration"]
    trees = X.export_trees(head.booster, names, rounds=rounds)
    row = E.design_row(design, names, 3)
    paths = X.tree_paths(trees, row, names)
    assert len(paths["trees"]) == rounds
    assert paths["total_path_length"] == sum(tree["path_length"] for tree in paths["trees"])
    margin = paths["margin_std"]
    assert abs(margin - sum(tree["leaf_value_std"] for tree in paths["trees"])) < 1e-9
    raw = margin * scaler["std"] + scaler["mean"]
    predicted = head.booster.predict(
        E.dmatrix(design[3:4], names), iteration_range=(0, rounds)
    )[0] * scaler["std"] + scaler["mean"]
    assert abs(raw - predicted) <= 1e-5


def test_case_selection_is_deterministic_and_distinct():
    target = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    pred_a = np.asarray([1.0, 2.5, 3.0, 4.4, 4.0])
    pred_f = np.asarray([1.2, 2.1, 3.2, 4.0, 5.1])
    rows = np.asarray([10, 11, 12, 13, 14])
    cases = E.select_cases(pred_a=pred_a, pred_f=pred_f, target=target, row_ids=rows)
    assert [case["kind"] for case in cases] == [
        "median_abs_error",
        "max_improvement_vs_F",
        "max_degradation_vs_F",
    ]
    assert len({case["row"] for case in cases}) == 3
    again = E.select_cases(pred_a=pred_a, pred_f=pred_f, target=target, row_ids=rows)
    assert [case["row"] for case in cases] == [case["row"] for case in again]


def test_witness_reports_distinct_branch_neighbours():
    graphs = [([6, 7, 6], [(0, 1), (1, 2)])]
    packed = make_packed(graphs)
    query = A.BRANCH(A.ATOM_EQ(0, 6), A.ATOM_EQ(0, 6), A.ANY_BOND_PRED(), A.ANY_BOND_PRED())
    assert A.evaluate(packed, query).tolist() == [False, True, False]
    wit = A.witness(packed, query, 1)
    assert wit["branch"]["u_neighbor"] != wit["branch"]["w_neighbor"]


def test_exported_query_json_is_executable_and_roundtrips():
    """queries.json -> AST -> counts must equal the in-memory query."""
    import sq_data as D
    import sq_explain as E

    cache = D.load_cache("ogbg-molesol")
    packed = D.packed_split(cache, "valid")
    vocab = D.train_vocabulary(cache)
    records, _ = B.build_base_bank(D.packed_split(cache, "train"), vocab, cache=B.MaskCache(limit_bytes=16 << 20))
    names = [f"{record.label}::{t}" for record in records[:1] for t in A.FEATURE_TRANSFORMS]
    payload = E.export_queries(
        records,
        dataset="ogbg-molesol",
        base_count=len(records),
        discovered_count=0,
        feature_names=names,
        scaler={"mean": 1.5, "std": 2.5},
        selected={"preset": "H1", "iteration": 3},
    )
    assert payload["feature_names"] == names
    assert payload["target_scaler"] == {"mean": 1.5, "std": 2.5}
    assert payload["selected_state"]["iteration"] == 3
    assert payload["n_queries"] == len(records)
    checked = 0
    for entry in payload["queries"]:
        ast = A.from_json(entry["ast_tree"])
        assert A.canon(ast) == entry["ast"]
        left = A.evaluate(packed, ast)
        right = A.evaluate(packed, A.from_json(E.ast_to_json(ast)))
        assert np.array_equal(left, right)
        assert entry["feature_columns"] == [f"{entry['label']}::{t}" for t in A.FEATURE_TRANSFORMS]
        checked += 1
    assert checked == len(records)


def test_exported_trees_replay_from_json_files(tmp_path):
    """Replay using only the two exported files (queries.json + native_trees.json)."""
    import json

    packed, design, names, scaler, y, head = _toy_model()
    rounds = head.best["iteration"]
    trees = X.export_trees(head.booster, names, rounds=rounds)
    path = tmp_path / "native_trees.json"
    path.write_text(json.dumps(trees), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    replay = X.replay_native(loaded, design, names, scaler)
    expected = head.booster.predict(E.dmatrix(design, names), iteration_range=(0, rounds)) * scaler["std"] + scaler["mean"]
    assert np.max(np.abs(replay - expected)) <= 1e-5
