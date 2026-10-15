"""Fast correctness tests for cscl-bjoc-signature-transfer-v1.

No labels, no checkpoints, no GPU, no official test.  Covers the molecule-cap
discovery table, percentile/tie handling, deterministic controls, the decision
rule, RDKit dataset-graph reconstruction, and the label-discipline tokens.
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

import bjoc_signature_transfer_v1 as bst  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/bjoc_signature_transfer_v1.yaml"


def _rec(seed: int, row: int, j: int, sig: tuple, ig: float) -> dict:
    return {"seed": seed, "row": row, "j": j, "signature": bst.sig_json(sig), "ig_abs": float(ig)}


X = (0, (0, 0), (1, 1), 1, 1)
Y = (2, (1, 1), (1, 2), 0, 0)


def test_config_frozen_constants():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == bst.PROTOCOL_ID
    assert cfg["discovery"]["frequency_floor_molecules"] == 20
    assert cfg["discovery"]["unknown_score_primary"] == 0.5
    assert cfg["validation"]["top_k_primary"] == 4
    assert cfg["validation"]["labels"] == "NEVER READ"
    assert "NEVER" in cfg["validation"]["official_test"]
    assert cfg["decision"]["labels"]["NO_SUPPORTED_SIGNATURE_TRANSFER"]


def test_average_rank_percentile_ties_and_singleton():
    p = bst.average_rank_percentile([3.0])
    assert p.tolist() == [0.5]
    p = bst.average_rank_percentile([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(p, [0.0, 1 / 3, 2 / 3, 1.0])
    p = bst.average_rank_percentile([1.0, 1.0, 2.0])
    assert p[0] == p[1] == 0.25 and p[2] == 1.0


def test_discovery_table_molecule_cap_controls_repeats():
    """A molecule with 10 copies of a class must not dominate a single instance."""
    records = []
    for j in range(10):
        records.append(_rec(0, 1, j, X, 10.0))
    for j in range(10, 20):
        records.append(_rec(0, 1, j, Y, 1.0))
    records.append(_rec(0, 2, 0, X, 10.0))
    records.append(_rec(0, 2, 1, Y, 1.0))
    records.append(_rec(0, 3, 0, (7, (1, 1), (1, 1), 0, 0), 5.0))
    table = bst.build_discovery_table(records, floor=2)
    xrow = [c for c in table["classes"] if bst.sig_key(c["signature"]) == X][0]
    # molecule cap: mean of the two molecule-level percentiles
    assert xrow["support_molecules"] == 2
    assert xrow["T"] == pytest.approx((0.7631578947368421 + 1.0) / 2.0, abs=1e-12)
    # junction-level weighting would have given a different value
    junction_level = (10 * 0.7631578947368421 + 1.0) / 11.0
    assert abs(xrow["T"] - junction_level) > 0.05
    assert any(c["support_molecules"] < 2 for c in table["classes"])


def test_candidate_floor_and_unknown_score():
    records = [
        _rec(0, 1, 0, X, 2.0),
        _rec(0, 1, 1, Y, 1.0),
        _rec(0, 2, 0, X, 2.0),
        _rec(0, 2, 1, Y, 1.0),
    ]
    table = bst.build_discovery_table(records, floor=2)
    assert table["n_candidates"] == 2
    values = bst.table_candidate_values(table)
    sigs = [X, Y, (5, (3, 3), (1, 1), 0, 0)]
    scores = bst.signature_scores(sigs, table, "primary")
    assert scores[2] == bst.UNKNOWN_SCORE_PRIMARY
    scores_min = bst.signature_scores(sigs, table, "min")
    assert scores_min[2] == pytest.approx(values.min())


def test_top_k_ties_stable_and_random_shuffle_determinism():
    scores = np.asarray([1.0, 1.0, 1.0, 0.0])
    assert bst.top_k_indices(scores, 2).tolist() == [0, 1]
    a = bst.random_draw_sets(10, 4, seed=7)
    b = bst.random_draw_sets(10, 4, seed=7)
    assert np.array_equal(a, b) and a.shape == (bst.RANDOM_DRAWS, 4)
    assert all(len(set(row.tolist())) == 4 for row in a)
    cand = np.asarray([0.1, 0.2, 0.3])
    codes = np.asarray([0, 1, -1, 2, 0])
    perm = np.asarray([2, 0, 1])
    top = bst.shuffle_top_sets(cand, codes, perm, 4)
    assert top.tolist() == [2, 0, 4, 3]  # unknown 0.5 first, ties keep junction order


def test_decision_rule_three_labels():
    all_pass = {k: True for k in ("M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")}
    assert bst.decide_from_gates(all_pass) == "TRANSFERABLE_SOURCE_SIGNATURE_SIGNAL"
    partial = dict(all_pass)
    partial["G2"] = False
    assert bst.decide_from_gates(partial) == "CONTEXT_DEPENDENT_OR_PARTIAL_TRANSFER"
    none = dict(all_pass)
    none["G1"] = False
    assert bst.decide_from_gates(none) == "NO_SUPPORTED_SIGNATURE_TRANSFER"
    low_cov = dict(all_pass)
    low_cov["M0"] = False
    assert bst.decide_from_gates(low_cov) == "NO_SUPPORTED_SIGNATURE_TRANSFER"


def test_rdkit_reconstruction_matches_object_counts():
    """Synthetic PyG-ZINC-style graph (explicit H words, kekulized ring)."""
    atom_types = [4, 4, 4, 4, 4, 4]  # 'C H1'
    bonds = [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1)]
    x = np.asarray(atom_types, dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    obj = sro.build_mol_objects(0, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))
    mol, tier = bst.build_rdkit_mol(x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))
    assert tier == "full"
    assert mol.GetNumAtoms() == int(obj.n_atoms) == 6
    assert mol.GetNumBonds() == int(obj.bond_index.shape[0]) == 6
    smi, scaffold, _t = bst.canonical_smiles_and_scaffold(x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))
    assert smi and scaffold
    arrs = bst.junction_arrays(obj)
    assert len(arrs["signatures"]) == int(obj.junc_index.shape[0])


def test_no_label_or_test_loader_tokens():
    module_text = (REPO_ROOT / "tracks/structure-contribution/code/bjoc_signature_transfer_v1.py").read_text(encoding="utf-8")
    runner_text = (REPO_ROOT / "tracks/structure-contribution/code/run_bjoc_signature_transfer_v1.py").read_text(encoding="utf-8")
    for text in (module_text, runner_text):
        assert "objects_test" not in text
        assert "locked_y" not in text
        assert "valid_y" not in text
        assert "monitor_read" not in text
    assert "official_test_loaded" in runner_text
    assert '"labels_read": False' in runner_text or "'labels_read': False" in runner_text


def test_runner_parser_commands_and_constants():
    import run_bjoc_signature_transfer_v1 as runner

    parser = runner.build_parser()
    commands = {a.dest: a.choices for a in parser._actions if a.dest == "cmd"}["cmd"]
    for name in ("verify-inputs", "freeze", "discovery", "structure", "validate", "analyze", "semantics", "finalize"):
        assert name in commands
    assert runner.bst.ALPHA == 1.0
    assert runner.bst.BOOTSTRAP_N == 5000
    assert runner.bst.RANDOM_RNG_BASE == 123_400_000
    assert runner.bst.SHUFFLE_RNG_BASE == 987_000_000
    assert runner.PHASE0_RANDOM_RNG_BASE == 70_000_000
