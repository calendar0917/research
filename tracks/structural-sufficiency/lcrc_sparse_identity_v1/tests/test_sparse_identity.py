"""Fast structural tests for the sparse-identity round (no training, CPU only).

Run with::

    uv run pytest -q tracks/structural-sufficiency/lcrc_sparse_identity_v1/tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import si_common as C  # noqa: E402
import si_features as FEAT  # noqa: E402
import si_model as M  # noqa: E402


# ---------------------------------------------------------------------------
# key codec
# ---------------------------------------------------------------------------


def test_key_codec_roundtrip():
    spec = FEAT.frozen_spec()
    rng = np.random.default_rng(0)
    roots = rng.integers(0, spec.n_atom_types, size=500)
    cats = rng.integers(0, spec.n_path_codes, size=500)
    ends = rng.integers(0, spec.n_nbr_codes, size=500)
    closures = rng.integers(0, 2, size=500)
    codes = FEAT.encode_key(roots, cats, ends, closures, spec)
    assert len(set(codes.tolist())) == 500
    for code, root, cat, end, closure in zip(codes.tolist(), roots, cats, ends, closures):
        assert FEAT.decode_key(code, spec) == (int(root), int(cat), int(end), int(closure))


def test_key_ordering_is_tuple_lexicographic():
    spec = FEAT.frozen_spec()
    a = FEAT.decode_key(int(FEAT.encode_key([1], [2], [3], [0], spec)[0]), spec)
    b = FEAT.decode_key(int(FEAT.encode_key([1], [2], [3], [1], spec)[0]), spec)
    c = FEAT.decode_key(int(FEAT.encode_key([1], [2], [4], [0], spec)[0]), spec)
    assert a < b < c


# ---------------------------------------------------------------------------
# features vs a small real cache subset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def subset():
    import ptb_data as PD
    import ptb_train_mol as PT

    view = PD.molecule_view(C.DATASET)
    rows = list(range(200))
    cache = PT._subset_cache(view.train_cache, rows, view.spec)
    return view.spec, view.train_norm, cache, view.train_cache


def test_path_keys_match_loop(subset):
    spec, _norm, cache, _full = subset
    records = FEAT.path_keys(cache, spec)
    for mol in range(int(cache["n_molecules"])):
        loop = sorted(FEAT.molecule_keys_loop(cache, spec, mol))
        sel = records["mol_of_path"] == mol
        vec = sorted(tuple(int(x) for x in FEAT.decode_key(int(code), spec)) for code in records["key"][sel])
        assert loop == vec


def test_document_frequency_counts_distinct_molecules(subset):
    spec, _norm, cache, _full = subset
    records = FEAT.path_keys(cache, spec)
    freq = FEAT.document_frequency(records)
    # brute force over molecules
    brute: dict[int, int] = {}
    for mol in range(int(cache["n_molecules"])):
        keys = {int(code) for code in records["key"][records["mol_of_path"] == mol].tolist()}
        for code in keys:
            brute[code] = brute.get(code, 0) + 1
    assert freq == brute
    assert sum(freq.values()) >= int(cache["n_molecules"]) - 0  # >= is always true; sanity only
    assert max(freq.values()) <= int(cache["n_molecules"])


def test_encode_split_normalisation_and_binary(subset):
    spec, _norm, cache, full = subset
    vocab = FEAT.build_vocabulary(full, spec, C.K)
    enc = FEAT.encode_split(cache, spec, vocab)
    x = enc["x"]
    binary = enc["binary"]
    active = binary.sum(axis=1)
    nonzero = active > 0
    assert np.allclose(np.linalg.norm(x[nonzero], axis=1), 1.0, atol=1e-6)
    assert np.all(x[~nonzero] == 0)
    assert np.all((binary == 0) | (binary == 1))
    for i in range(0, int(cache["n_molecules"]), 37):
        expected = binary[i] / np.sqrt(max(1, int(active[i])))
        assert np.allclose(x[i], expected, atol=1e-7)
    # molecule with no selected pattern stays all-zero
    empty = np.flatnonzero(active == 0)
    if empty.size:
        assert not x[empty[0]].any()
    assert enc["stats"]["n_nonzero"] == int(binary.sum())


def test_encoding_is_order_invariant_on_real_molecule(subset):
    spec, _norm, _cache, _full = subset
    vocab = FEAT.build_vocabulary(_full, spec, C.K)
    import mb2_rdkit as RDK

    from rdkit import Chem
    import lcrc_data as PREP_DATA

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin, real ring/aromatic content
    raw = RDK.smiles_to_raw_row(smiles, 0)
    base = PREP_DATA.build_molecule(RDK.raw_row_to_graph(raw, 0, spec), spec)
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    ren = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    raw2 = RDK._mol_to_raw_row(ren, Chem.MolToSmiles(ren), 0)
    other = PREP_DATA.build_molecule(RDK.raw_row_to_graph(raw2, 0, spec), spec)
    v0 = FEAT.encode_split(PREP_DATA.concat_molecules([base], spec, split="a"), spec, vocab)["x"][0]
    v1 = FEAT.encode_split(PREP_DATA.concat_molecules([other], spec, split="b"), spec, vocab)["x"][0]
    assert np.array_equal(v0, v1)
    # reversing each root's path block keeps the vector
    rev = dict(base)
    path_ptr = np.asarray(base["path_ptr"])
    order = []
    for atom in range(int(base["n_atoms"])):
        order.extend(range(int(path_ptr[atom]), int(path_ptr[atom + 1]))[::-1])
    for field in ("path_cat", "path_end63", "path_end", "path_closure"):
        rev[field] = np.asarray(base[field])[order]
    v2 = FEAT.encode_split(PREP_DATA.concat_molecules([rev], spec, split="c"), spec, vocab)["x"][0]
    assert np.array_equal(v0, v2)


# ---------------------------------------------------------------------------
# NULL randomisation
# ---------------------------------------------------------------------------


def test_double_edge_swap_conserves_marginals_and_changes_matrix():
    rng = np.random.default_rng(7)
    mat = (rng.random((300, 40)) < 0.08).astype(np.uint8)
    null, stats = FEAT.double_edge_swap(mat, 20261103)
    assert np.array_equal(mat.sum(axis=1), null.sum(axis=1))
    assert np.array_equal(mat.sum(axis=0), null.sum(axis=0))
    assert int(mat.sum()) == int(null.sum())
    assert not np.array_equal(mat, null)
    assert stats["n_attempts"] == C.NULL_SWAP_MULTIPLIER * int(mat.sum())
    again, _ = FEAT.double_edge_swap(mat, 20261103)
    assert np.array_equal(null, again)
    other, _ = FEAT.double_edge_swap(mat, 20261104)
    assert not np.array_equal(null, other)


def test_double_edge_swap_rejects_degenerate():
    mat = np.zeros((5, 3), dtype=np.uint8)
    null, stats = FEAT.double_edge_swap(mat, 1)
    assert stats["n_attempts"] == 0
    assert np.array_equal(mat, null)


# ---------------------------------------------------------------------------
# FOLD64
# ---------------------------------------------------------------------------


def _synthetic_vocab(n_keys: int = 512) -> dict:
    # unique keys inside the real code ranges
    seen = set()
    out = []
    i = 0
    while len(out) < n_keys:
        key = (i % 12, (i // 12) % 400, (i // 4800) % 100, i % 2)
        if key not in seen:
            seen.add(key)
            out.append([int(x) for x in key])
        i += 1
    return {"keys": out, "k": n_keys}


def test_fold_buckets_partition():
    vocab = _synthetic_vocab()
    fold = FEAT.fold_buckets(vocab)
    b = np.asarray(fold["bucket_of_col"])
    assert b.size == 512
    counts = np.bincount(b, minlength=64)
    assert counts.tolist() == [8] * 64
    assert len(set(fold["digest_prefix"])) == 512
    again = FEAT.fold_buckets(vocab)
    assert np.array_equal(b, np.asarray(again["bucket_of_col"]))


def test_fold_projection_formula():
    rng = np.random.default_rng(3)
    x = (rng.random((7, 512)) < 0.1).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    buckets = np.arange(512) % 64
    z = FEAT.project_fold64(x, buckets)
    assert z.shape == (7, 64)
    manual = np.zeros_like(z)
    for j in range(64):
        cols = np.flatnonzero(buckets == j)
        manual[:, j] = x[:, cols].sum(axis=1) / np.sqrt(8.0)
    assert np.allclose(z, manual, atol=1e-6)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec_norm():
    import ptb_data as PD

    view = PD.molecule_view(C.DATASET)
    return view.spec, view.train_norm


def test_parameter_counts(spec_norm):
    spec, norm = spec_norm
    a0 = C.PC.EXPECTED_PARAMS_SINGLE[C.MODEL_A0]
    counts = {}
    for arm in C.NEURAL:
        model, info = M.build_model(arm, spec, norm, 0, "cpu")
        counts[arm] = int(sum(p.numel() for p in model.parameters()))
        assert counts[arm] == int(info["parameters"])
    assert counts[C.MODEL_A0] == a0 == 247106
    assert counts[C.MODEL_TRUE] == counts[C.MODEL_NULL] == a0 + 512
    assert counts[C.MODEL_FOLD] == a0 + 64


def test_wide_trunk_bitwise_equal_and_zero_branch(spec_norm):
    spec, norm = spec_norm
    a0, _ = M.build_model(C.MODEL_A0, spec, norm, 1, "cpu")
    wide, _ = M.build_model(C.MODEL_TRUE, spec, norm, 1, "cpu")
    a0_state = a0.state_dict()
    wide_state = wide.state_dict()
    for key, value in a0_state.items():
        assert torch.equal(value, wide_state[f"base.{key}"])
    assert float(wide.branch.weight.abs().sum()) == 0.0


def test_wide_branch_does_not_consume_global_rng(spec_norm):
    spec, norm = spec_norm
    torch.manual_seed(123)
    M.build_model(C.MODEL_A0, spec, norm, 0, "cpu")
    after_a0 = torch.get_rng_state().clone()
    torch.manual_seed(123)
    M.build_model(C.MODEL_TRUE, spec, norm, 0, "cpu")
    after_wide = torch.get_rng_state().clone()
    assert torch.equal(after_a0, after_wide)


def test_branch_receives_gradient(spec_norm):
    spec, norm = spec_norm
    model, _ = M.build_model(C.MODEL_TRUE, spec, norm, 0, "cpu")
    feat = torch.zeros(4, 512)
    feat[:, 3] = 1.0
    # forward through the branch alone (the A0 part is exercised elsewhere)
    out = model.branch(feat).reshape(-1)
    out.sum().backward()
    assert float(model.branch.weight.grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# recorded artefacts (present after the gate run)
# ---------------------------------------------------------------------------


def test_gate_artifacts_are_consistent():
    analysis = CODE.parent / "analysis"
    vocab_path = analysis / "feature_vocabulary.json"
    if not vocab_path.exists():
        pytest.skip("gate artefacts not generated yet")
    vocab = json.loads(vocab_path.read_text())
    assert vocab["status"] == "OK"
    assert vocab["k"] == 512
    assert len(vocab["keys"]) == 512 == len(vocab["document_frequency"])
    assert vocab["vocabulary_sha256"] == FEAT.vocabulary_hash(vocab["keys"], vocab["document_frequency"])
    ordered = sorted(
        zip(vocab["keys"], vocab["document_frequency"]),
        key=lambda kv: (-kv[1], tuple(kv[0])),
    )
    assert [list(k) for k, _ in ordered] == vocab["keys"]
    gates = json.loads((analysis / "correctness_gates.json").read_text())
    for gate in gates["gates"]:
        if gate["id"] in (13, 14) and gate.get("pass") is None:
            continue
        assert gate["pass"] is True, gate
